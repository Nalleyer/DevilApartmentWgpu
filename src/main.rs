use std::f32::consts::PI;
use std::{borrow::Cow, mem};

use bytemuck_derive::{Pod, Zeroable};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupLayoutEntry, BufferUsages, CommandEncoderDescriptor, Extent3d,
    ShaderModuleDescriptor, ShaderStages, SurfaceConfiguration, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
};
use wgpu::{Color, RenderPassColorAttachment, RenderPassDescriptor};
use winit::dpi::PhysicalSize;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 4],
    uv: [f32; 2],
}

impl Vertex {
    fn new(pos: [i8; 2], uv: [f32; 2]) -> Self {
        Self {
            pos: [pos[0] as f32, pos[1] as f32, 0.0, 1.0],
            uv,
        }
    }

    // uv
    // +
    // |  \
    // |     \
    // +------+
    // |      |  \
    // |      |    \
    // +------+------+
    pub fn new_big_triangle() -> (Vec<Self>, Vec<u16>) {
        let vertices = vec![
            Self::new([-1, 1], [0.0, 0.0]),
            Self::new([-1, -1], [0.0, 1.0]),
            Self::new([1, -1], [1.0, 1.0]),
            Self::new([1, 1], [1.0, 0.0]),
        ];
        // use right hand normal here
        let indices = vec![0, 1, 2, 0, 2, 3];

        (vertices, indices)
    }
}

const COLOR_BG: [u8; 3] = [255, 0, 0];
const COLOR1: [u8; 3] = [175, 211, 105];
const COLOR2: [u8; 3] = [206, 234, 247];
const COLOR3: [u8; 3] = [204, 215, 228];
const COLOR4: [u8; 3] = [213, 201, 223];
const COLOR5: [u8; 3] = [220, 184, 203];
const COLOR6: [u8; 3] = [33, 33, 33];

fn get_color(row: usize, col: usize, size: usize) -> [u8; 3] {
    let one_tree: f32 = 1.0 / 3.0;
    let two_tree: f32 = 2.0 / 3.0;
    let xf = (col as f32) / (size as f32);
    let yf = (row as f32) / (size as f32);
    let delta = 0.05f32;
    if xf <= 0.25 {
        if yf >= (one_tree - delta) && yf <= (two_tree + delta) {
            COLOR2
        } else {
            COLOR_BG
        }
    } else if xf <= 0.5 {
        if yf <= one_tree {
            COLOR1
        } else if yf <= two_tree {
            COLOR3
        } else {
            COLOR6
        }
    } else if xf <= 0.75 {
        if yf >= (one_tree - delta) && yf <= (two_tree + delta) {
            COLOR4
        } else {
            COLOR_BG
        }
    } else {
        if yf >= (one_tree - delta) && yf <= (two_tree + delta) {
            COLOR5
        } else {
            COLOR_BG
        }
    }
}

// format rgb8
fn create_texels(size: usize) -> Vec<u8> {
    let mut result = vec![0u8; size * size * 4];
    for i in 0..size * size {
        let row = i / size;
        let col = i % size;
        let color = get_color(row, col, size);
        result[i * 4] = color[0];
        result[i * 4 + 1] = color[1];
        result[i * 4 + 2] = color[2];
        result[i * 4 + 3] = 255;
    }
    result
}

struct App {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: usize,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    surface_config: SurfaceConfiguration,
    vertices: Vec<Vertex>,
    theta: f32,
}

impl App {
    pub async fn new(window: &Window) -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let size = window.inner_size();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let surface_texture_format = surface.get_supported_formats(&adapter)[0];

        let adapter_features = adapter.features();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: adapter_features,
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_texture_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
        };
        let (vertices, indices) = Vertex::new_big_triangle();

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            //border_color: Some(SamplerBorderColor::TransparentBlack),
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let size = 4096;
        let texels = create_texels(size);
        let texture_extent = Extent3d {
            width: size as u32,
            height: size as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        });
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        queue.write_texture(
            texture.as_image_copy(),
            &texels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new((size * 4) as u32).unwrap()),
                rows_per_image: None,
            },
            texture_extent,
        );

        //let aspect_ratio = (surface_config.width as f32) / (surface_config.height as f32);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shader.wgsl"))),
        });

        let vertex_size = mem::size_of::<Vertex>();
        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 4 * 4,
                    shader_location: 1,
                },
            ],
        }];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            device,
            queue,
            surface,
            bind_group,
            vertex_buffer,
            index_buffer,
            index_count: indices.len(),
            pipeline,
            surface_config,
            vertices,
            theta: 0.0,
        }
    }

    fn update_surface_config(&mut self, width: u32, height: u32) {
        self.surface_config.height = height;
        self.surface_config.width = width;
    }

    pub fn resize(&mut self, size: &PhysicalSize<u32>) {
        println!("size: {:?}", size);
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.update_surface_config(size.width, size.height);
        self.surface.configure(&self.device, &self.surface_config);

        let (w, h) = (self.surface_config.width, self.surface_config.height);

        if w >= h {
            let height_to_width = (h as f32) / (w as f32);
            // change x
            self.vertices[0].pos[0] = -1.0 * height_to_width;
            self.vertices[1].pos[0] = -1.0 * height_to_width;
            self.vertices[2].pos[0] = 1.0 * height_to_width;
            self.vertices[3].pos[0] = 1.0 * height_to_width;
            // normal y
            self.vertices[0].pos[1] = 1.0;
            self.vertices[1].pos[1] = -1.0;
            self.vertices[2].pos[1] = -1.0;
            self.vertices[3].pos[1] = 1.0;
        } else {
            let width_to_height = (w as f32) / (h as f32);
            // change y
            self.vertices[0].pos[1] = 1.0 * width_to_height;
            self.vertices[1].pos[1] = -1.0 * width_to_height;
            self.vertices[2].pos[1] = -1.0 * width_to_height;
            self.vertices[3].pos[1] = 1.0 * width_to_height;
            // normal x
            self.vertices[0].pos[0] = -1.0;
            self.vertices[1].pos[0] = -1.0;
            self.vertices[2].pos[0] = 1.0;
            self.vertices[3].pos[0] = 1.0;
        }

        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
    }

    pub fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                self.surface.configure(&self.device, &self.surface_config);
                self.surface
                    .get_current_texture()
                    .expect("Failed to acquire next surface texture!")
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // rp
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            rpass.push_debug_group("Prepare data for draw.");
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.pop_debug_group();
            rpass.insert_debug_marker("Draw!");
            rpass.draw_indexed(0..self.index_count as u32, 0, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));

        frame.present();
    }

    pub fn update(&mut self) {
        self.theta += 0.001;
        if self.theta > 2.0 * PI {
            self.theta -= 2.0 * PI;
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let mut app = pollster::block_on(App::new(&window));
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            }
            | WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
            }
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(VirtualKeyCode::R),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {}
            WindowEvent::Resized(size)
            | WindowEvent::ScaleFactorChanged {
                new_inner_size: &mut size,
                ..
            } => {
                app.resize(&size);
            }
            _ => (),
        },
        Event::RedrawEventsCleared => {
            app.update();
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            app.render();
        }
        _ => (),
    });
}
