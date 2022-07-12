struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
) -> VertexOutput {
    var result: VertexOutput;
    result.tex_coord = tex_coord;
    result.position = position;
    return result;
}

@group(0)
@binding(0)
var r_color: texture_2d<f32>;

@group(0)@binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let tex = textureSample(r_color, s_diffuse, vertex.tex_coord);
    return tex;
}
