//! Pure assembly of normalized Precision Touchpad hybrid reports.
//!
//! The first fragment declares the frame's contact count; continuations have a zero count
//! and the same scan time. A zero count with no contacts and no pending frame is all-up.
//! Scan times are opaque device counters (including wrap); expiry uses caller monotonic time.
//! This module neither registers Windows input nor suppresses or forwards any input.

/// Protocol ceiling, independent of a descriptor's contact-ID logical range.
pub(crate) const MAX_CONTACTS: usize = 16;

/// One normalized contact. Missing optional fields remain missing, including pressure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Contact {
    pub id: u32,
    pub position: Option<(i32, i32)>,
    pub tip: bool,
    pub pressure: Option<u16>,
    pub width: Option<u16>,
    pub height: Option<u16>,
}

/// Borrowed decoder output; assembly never retains this potentially unbounded slice.
pub(crate) struct Fragment<'a> {
    pub scan_time: Option<u32>,
    pub contact_count: Option<usize>,
    /// None means this fragment does not carry device-wide button state.
    pub buttons: Option<u32>,
    pub contacts: &'a [Contact],
}

/// A complete bounded frame. Tip=false entries preserve explicit contact lifts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Frame {
    pub scan_time: u32,
    pub buttons: u32,
    contacts: [Contact; MAX_CONTACTS],
    len: usize,
}

impl Frame {
    pub fn contacts(&self) -> &[Contact] {
        &self.contacts[..self.len]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Error {
    Metadata,
    Count,
    MissingStart,
    DuplicateStart,
    ScanChanged,
    MissingPosition,
    DuplicateId,
    Expired,
    ClockReversed,
}

/// A fixed-size accumulator. Errors clear pending state and must trigger caller recovery.
pub(crate) struct Assembler {
    max_contacts: usize,
    timeout_ms: u64,
    pending: Option<Frame>,
    expected: usize,
    started_ms: u64,
    last_ms: Option<u64>,
    buttons: u32,
}

impl Assembler {
    pub fn new(max_contacts: usize, timeout_ms: u64) -> Result<Self, Error> {
        if !(1..=MAX_CONTACTS).contains(&max_contacts) || timeout_ms == 0 {
            return Err(Error::Count);
        }
        Ok(Self {
            max_contacts,
            timeout_ms,
            pending: None,
            expected: 0,
            started_ms: 0,
            last_ms: None,
            buttons: 0,
        })
    }

    /// Discard partial data and button state on device removal or capture-session change.
    pub fn reset(&mut self) {
        self.pending = None;
        self.expected = 0;
        self.started_ms = 0;
        self.last_ms = None;
        self.buttons = 0;
    }

    /// Call even when no reports arrive. A timeout never fabricates a complete frame.
    pub fn expire(&mut self, now_ms: u64) -> Result<(), Error> {
        let error = if self.last_ms.is_some_and(|last| now_ms < last) {
            Some(Error::ClockReversed)
        } else if self.pending.is_some() && now_ms - self.started_ms >= self.timeout_ms {
            Some(Error::Expired)
        } else {
            None
        };
        if let Some(error) = error {
            self.reset();
            return Err(error);
        }
        self.last_ms = Some(now_ms);
        Ok(())
    }

    /// Returns None until every declared record arrives; an error consumes no partial frame.
    pub fn push(&mut self, fragment: Fragment<'_>, now_ms: u64) -> Result<Option<Frame>, Error> {
        self.expire(now_ms)?;
        let result = self.push_checked(fragment, now_ms);
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn push_checked(
        &mut self,
        fragment: Fragment<'_>,
        now_ms: u64,
    ) -> Result<Option<Frame>, Error> {
        let scan_time = fragment.scan_time.ok_or(Error::Metadata)?;
        let count = fragment.contact_count.ok_or(Error::Metadata)?;
        if count > self.max_contacts || fragment.contacts.len() > self.max_contacts {
            return Err(Error::Count);
        }
        if let Some(frame) = &self.pending {
            if frame.scan_time != scan_time {
                return Err(Error::ScanChanged);
            }
            if count != 0 {
                return Err(Error::DuplicateStart);
            }
            if fragment.contacts.is_empty() {
                return Err(Error::Count);
            }
        } else {
            if count == 0 && !fragment.contacts.is_empty() {
                return Err(Error::MissingStart);
            }
            if count != 0 && fragment.contacts.is_empty() {
                return Err(Error::Count);
            }
            self.expected = count;
            self.started_ms = now_ms;
            self.pending = Some(Frame {
                scan_time,
                buttons: self.buttons,
                contacts: [Contact::default(); MAX_CONTACTS],
                len: 0,
            });
        }
        let frame = self.pending.as_mut().expect("frame initialized above");
        if frame.len + fragment.contacts.len() > self.expected {
            return Err(Error::Count);
        }
        if let Some(buttons) = fragment.buttons {
            frame.buttons = buttons;
        }
        for contact in fragment.contacts {
            if contact.position.is_none() {
                return Err(Error::MissingPosition);
            }
            if frame
                .contacts()
                .iter()
                .any(|previous| previous.id == contact.id)
            {
                return Err(Error::DuplicateId);
            }
            frame.contacts[frame.len] = *contact;
            frame.len += 1;
        }
        if frame.len == self.expected {
            self.buttons = frame.buttons;
            return Ok(self.pending.take());
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contact(id: u32) -> Contact {
        Contact {
            id,
            position: Some((1717, 911)),
            tip: true,
            ..Contact::default()
        }
    }

    fn part(scan: u32, count: usize, contacts: &[Contact]) -> Fragment<'_> {
        Fragment {
            scan_time: Some(scan),
            contact_count: Some(count),
            buttons: None,
            contacts,
        }
    }

    #[test]
    fn one_through_five_and_all_up() {
        let mut a = Assembler::new(5, 20).unwrap();
        for count in 1..=5 {
            for id in 0..count {
                let contacts = [contact(id as u32)];
                let result = a
                    .push(
                        part(count as u32, if id == 0 { count } else { 0 }, &contacts),
                        count as u64,
                    )
                    .unwrap();
                if id + 1 == count {
                    let frame = result.unwrap();
                    assert_eq!(frame.contacts().len(), count);
                    assert_eq!(frame.contacts()[id].id, id as u32);
                    assert_eq!(frame.contacts()[id].pressure, None);
                } else {
                    assert!(result.is_none());
                }
            }
        }
        assert!(
            a.push(part(6, 0, &[]), 6)
                .unwrap()
                .unwrap()
                .contacts()
                .is_empty()
        );
    }

    #[test]
    fn optional_fields_lift_buttons_and_counter_wrap() {
        let mut a = Assembler::new(5, 20).unwrap();
        let first = [Contact {
            pressure: Some(0),
            width: Some(8),
            height: Some(9),
            ..contact(4)
        }];
        let mut fragment = part(u32::MAX, 2, &first);
        fragment.buttons = Some(1);
        assert!(a.push(fragment, 0).unwrap().is_none());
        let lift = [Contact {
            tip: false,
            ..contact(0)
        }];
        let frame = a.push(part(u32::MAX, 0, &lift), 1).unwrap().unwrap();
        assert_eq!(frame.buttons, 1);
        assert_eq!(frame.contacts()[0], first[0]);
        assert!(!frame.contacts()[1].tip);
        let mut up = part(0, 0, &[]);
        up.buttons = Some(0);
        assert_eq!(a.push(up, 2).unwrap().unwrap().buttons, 0);
    }

    #[test]
    fn malformed_fragments_clear_pending() {
        let mut a = Assembler::new(5, 20).unwrap();
        let contacts = [contact(0)];
        assert_eq!(a.push(part(1, 0, &contacts), 0), Err(Error::MissingStart));
        assert!(a.push(part(1, 2, &contacts), 1).unwrap().is_none());
        assert_eq!(a.push(part(1, 0, &contacts), 2), Err(Error::DuplicateId));
        assert!(a.push(part(2, 2, &contacts), 3).unwrap().is_none());
        assert_eq!(a.push(part(3, 0, &contacts), 4), Err(Error::ScanChanged));
        assert!(a.push(part(4, 2, &contacts), 5).unwrap().is_none());
        assert_eq!(a.push(part(4, 2, &contacts), 6), Err(Error::DuplicateStart));
        assert_eq!(a.push(part(5, 6, &contacts), 7), Err(Error::Count));
        let missing = [Contact {
            position: None,
            ..contact(0)
        }];
        assert_eq!(a.push(part(6, 1, &missing), 8), Err(Error::MissingPosition));
        let mut fragment = part(7, 1, &contacts);
        fragment.scan_time = None;
        assert_eq!(a.push(fragment, 9), Err(Error::Metadata));
        assert!(a.pending.is_none());
    }

    #[test]
    fn expiry_reset_and_backward_clock() {
        let mut a = Assembler::new(5, 20).unwrap();
        let contacts = [contact(0)];
        a.push(part(1, 2, &contacts), 100).unwrap();
        assert_eq!(a.expire(120), Err(Error::Expired));
        assert_eq!(a.push(part(1, 0, &contacts), 121), Err(Error::MissingStart));
        a.push(part(2, 2, &contacts), 200).unwrap();
        assert_eq!(a.expire(199), Err(Error::ClockReversed));
        a.push(part(3, 2, &contacts), 300).unwrap();
        a.reset();
        assert!(
            a.push(part(4, 0, &[]), 0)
                .unwrap()
                .unwrap()
                .contacts()
                .is_empty()
        );
        assert!(Assembler::new(0, 20).is_err());
        assert!(Assembler::new(MAX_CONTACTS + 1, 20).is_err());
        assert!(Assembler::new(5, 0).is_err());
    }
}
