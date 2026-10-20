// The display's present shaders and the XR10 paging kernels, compiled at run time once per Metal
// device (gpu/shaders.rs). Core binds their arguments: display/render.rs for the present pass,
// resources/texture/compute.rs for the XR10 dispatches.

#include <metal_stdlib>
using namespace metal;

// ---- The display ----

// A vertex of the full-screen strip: its clip-space position and the frame's texture coordinate.
struct PresentVertex {
    float4 position [[position]];
    float2 tex;
};

// Vertex `vid` of the strip: the display's position buffer (float4 each) at 0 and its
// texture-coordinate buffer (float2 each) at 1.
vertex PresentVertex displayPresentVertex(constant float4 *positions [[buffer(0)]],
                                          constant float2 *tex_coords [[buffer(1)]],
                                          uint vid [[vertex_id]]) {
    PresentVertex out;
    out.position = positions[vid];
    out.tex = tex_coords[vid];
    return out;
}

// The frame, filtered where the host's region scales it.
constexpr sampler frame_sampler(coord::normalized, address::clamp_to_edge, filter::linear);

// Without a gamma table: the frame as it is, alpha included. The colour is neither
// premultiplied nor blended, and the render target's format converts it.
fragment float4 displayPresentFragment(PresentVertex in [[stage_in]],
                                       texture2d<float> frame [[texture(0)]]) {
    return frame.sample(frame_sampler, in.tex);
}

// The display's fragment property buffer (96 bytes, written by display/render.rs): the colour
// matrix as four columns of three floats, each column padded to 16 bytes, then the gamma table's
// slopes at 0 and at 1, three packed floats each.
struct DisplayFragmentProperties {
    float4x3 color_matrix;
    packed_float3 gamma_slope_at_zero;
    packed_float3 gamma_slope_at_one;
};

static_assert(sizeof(DisplayFragmentProperties) == 96, "the property buffer's length");

// The gamma table's entries, one slice per channel: the entry whose span holds the value, not
// an interpolation between entries (a two-entry ramp is a threshold at 0.5).
constexpr sampler gamma_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);

// Channel `channel` of `v` through its table. Inside [0, 1] the table's entry; outside it the
// table's end entry, extended by the table's slope at that end, since nothing clamps a surface's
// components (an 'RGhA' frame holds values below 0 and above 1).
static float gamma_lookup(texture1d_array<float> table, uint channel, float v, float slope_at_zero,
                          float slope_at_one) {
    float inside = table.sample(gamma_sampler, clamp(v, 0.0f, 1.0f), channel).r;
    float at_zero = table.sample(gamma_sampler, 0.0f, channel).r;
    float at_one = table.sample(gamma_sampler, 1.0f, channel).r;
    if (v < 0.0f) {
        return at_zero + v * slope_at_zero;
    }
    if (v > 1.0f) {
        return at_one + (v - 1.0f) * slope_at_one;
    }
    return inside;
}

// With a gamma table: each channel through its table, then row c of the colour matrix gives
// output c = m[c][0] r + m[c][1] g + m[c][2] b + m[c][3]. Preserves the frame's sampled alpha.
fragment float4 displayPresentFragmentWithGammaTableAndColorMatrix(
    PresentVertex in [[stage_in]], texture2d<float> frame [[texture(0)]],
    texture1d_array<float> gamma_table [[texture(1)]],
    constant DisplayFragmentProperties &properties [[buffer(0)]]) {
    float4 sampled = frame.sample(frame_sampler, in.tex);
    float3 color = sampled.rgb;
    float3 slope_at_zero = properties.gamma_slope_at_zero;
    float3 slope_at_one = properties.gamma_slope_at_one;
    float3 looked_up;
    for (uint c = 0; c < 3; c++) {
        looked_up[c] = gamma_lookup(gamma_table, c, color[c], slope_at_zero[c], slope_at_one[c]);
    }
    return float4(properties.color_matrix * float4(looked_up, 1.0f), sampled.a);
}

// ---- XR10 two-plane textures (pixel formats 550 and 551) ----
//
// The guest's texture is two planes: colour, viewed as BGR10_XR (or its sRGB form), and alpha,
// viewed in a four-byte format whose first component is the byte after the texel's first. The
// kernels move texels between the planes and the host's texture, one thread per texel.

// Paging in: colour from plane 0 and alpha from plane 1, into the texture.
kernel void BlitInRGB_2P_XR10_A8(texture2d<half, access::read> src_color [[texture(0)]],
                                 texture2d<half, access::read> src_alpha [[texture(1)]],
                                 texture2d<half, access::write> dst [[texture(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    dst.write(half4(src_color.read(gid).rgb, src_alpha.read(gid).r), gid);
}

// Paging out: the texture's colour into plane 0 and its alpha into every component of plane 1.
kernel void BlitOutRGB_2P_XR10_A8(texture2d<half, access::read> src [[texture(0)]],
                                  texture2d<half, access::write> dst_color [[texture(1)]],
                                  texture2d<half, access::write> dst_alpha [[texture(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    half4 texel = src.read(gid);
    dst_color.write(half4(texel.rgb, 1.0h), gid);
    dst_alpha.write(half4(texel.a), gid);
}
