use std::{borrow::Cow, mem};

use bytemuck_derive::{Pod, Zeroable};

use egui::FontDefinitions;
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::{Platform, PlatformDescriptor};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupLayoutEntry, BufferUsages, CommandEncoderDescriptor, Extent3d,
    FilterMode, ShaderModuleDescriptor, ShaderStages, SurfaceConfiguration, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
};
use wgpu::{Color, RenderPassColorAttachment, RenderPassDescriptor};
use winit::dpi::PhysicalSize;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const DISPLAY_WIDTH: u32 = 426;
const DISPLAY_HEIGHT: u32 = 240;

struct EguiContext {
    render_pass: RenderPass,
    platform: Platform,
}

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

    pub fn new_rect() -> (Vec<Self>, Vec<u16>) {
        let vertices = vec![
            Self::new([-1, 1], [0.0, 0.0]),
            Self::new([-1, -1], [0.0, 1.0]),
            Self::new([1, -1], [1.0, 1.0]),
            Self::new([1, 1], [1.0, 0.0]),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        (vertices, indices)
    }
}

const COLOR_TEST: [u8; 3] = [100, 100, 0];

// format rgb8
fn create_texels() -> Vec<u8> {
    let wh = (DISPLAY_HEIGHT * DISPLAY_WIDTH) as usize;
    let mut result = vec![0u8; wh * 4];
    for i in 0..wh {
        let x: u32 = (i as u32) % DISPLAY_WIDTH;
        let percent = (x as f32) / (DISPLAY_WIDTH as f32);
        let color = COLOR_TEST;
        result[i * 4] = (256.0 * percent) as u8;
        result[i * 4 + 1] = color[1];
        result[i * 4 + 2] = color[2];
        result[i * 4 + 3] = 255;
    }
    result
}

fn get_length_in_pixel(screen_length: u32, display_length: u32) -> u32 {
    debug_assert!(display_length <= screen_length);
    let mut scale = 1;
    while scale * display_length * 2 < screen_length {
        scale *= 2;
    }
    display_length * scale
}

struct App {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface,
    texture: wgpu::Texture,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: usize,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    surface_config: SurfaceConfiguration,
    surface_texture_format: TextureFormat,
    vertices: Vec<Vertex>,
    egui_context: EguiContext,
    texels: Vec<u8>,
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
            present_mode: wgpu::PresentMode::AutoNoVsync,
        };
        let (vertices, indices) = Vertex::new_rect();

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
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
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

        let texels = create_texels();
        let texture_extent = Extent3d {
            width: DISPLAY_WIDTH,
            height: DISPLAY_HEIGHT,
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
                bytes_per_row: Some(std::num::NonZeroU32::new(DISPLAY_WIDTH * 4u32).unwrap()),
                rows_per_image: None,
            },
            texture_extent,
        );

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

        let platform = Platform::new(PlatformDescriptor {
            physical_width: window.inner_size().width,
            physical_height: window.inner_size().height,
            scale_factor: window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: Default::default(),
        });

        let egui_rpass = RenderPass::new(&device, surface_texture_format, 1);

        let egui_context = EguiContext {
            render_pass: egui_rpass,
            platform,
        };

        Self {
            device,
            texels,
            queue,
            surface,
            texture,
            bind_group,
            vertex_buffer,
            index_buffer,
            index_count: indices.len(),
            pipeline,
            surface_config,
            vertices,
            surface_texture_format,
            egui_context,
        }
    }

    fn update_surface_config(&mut self, width: u32, height: u32) {
        self.surface_config.height = height;
        self.surface_config.width = width;
    }

    pub fn resize(&mut self, size: &PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.update_surface_config(size.width, size.height);
        self.surface.configure(&self.device, &self.surface_config);

        let (w, h) = (self.surface_config.width, self.surface_config.height);

        let (height_percent, width_percent) = if w >= h {
            let height = get_length_in_pixel(h, DISPLAY_HEIGHT);
            let width = height / DISPLAY_HEIGHT * DISPLAY_WIDTH;
            let height_percent = (height as f32) / (h as f32);
            let width_percent = (width as f32) / (w as f32);
            (height_percent, width_percent)
        } else {
            let width = get_length_in_pixel(w, DISPLAY_WIDTH);
            let height = width / DISPLAY_WIDTH * DISPLAY_HEIGHT;
            let height_percent = (height as f32) / (h as f32);
            let width_percent = (width as f32) / (w as f32);
            (height_percent, width_percent)
        };

        self.vertices[0].pos[0] = -1.0 * width_percent;
        self.vertices[1].pos[0] = -1.0 * width_percent;
        self.vertices[2].pos[0] = 1.0 * width_percent;
        self.vertices[3].pos[0] = 1.0 * width_percent;
        self.vertices[0].pos[1] = 1.0 * height_percent;
        self.vertices[1].pos[1] = -1.0 * height_percent;
        self.vertices[2].pos[1] = -1.0 * height_percent;
        self.vertices[3].pos[1] = 1.0 * height_percent;

        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
    }

    pub fn render(&mut self, window: &Window) {
        let texture_extent = Extent3d {
            width: DISPLAY_WIDTH,
            height: DISPLAY_HEIGHT,
            depth_or_array_layers: 1,
        };
        // test write texture

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &self.texels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(DISPLAY_WIDTH * 4u32).unwrap()),
                rows_per_image: None,
            },
            texture_extent,
        );

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

        let screen_descriptor = ScreenDescriptor {
            physical_width: self.surface_config.width,
            physical_height: self.surface_config.height,
            scale_factor: window.scale_factor() as f32,
        };

        self.egui_context.platform.begin_frame();

        // all ui logic here
        egui::Window::new("fps").show(&self.egui_context.platform.context(), |ui| {
            ui.label("fps: unknown");
        });

        let full_output = self.egui_context.platform.end_frame(Some(window));
        let paint_jobs = self
            .egui_context
            .platform
            .context()
            .tessellate(full_output.shapes);

        let tdelta: egui::TexturesDelta = full_output.textures_delta;
        self.egui_context
            .render_pass
            .add_textures(&self.device, &self.queue, &tdelta)
            .expect("add texture ok");

        self.egui_context.render_pass.update_buffers(
            &self.device,
            &self.queue,
            &paint_jobs,
            &screen_descriptor,
        );

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
            self.egui_context
                .render_pass
                .execute_with_renderpass(&mut rpass, &paint_jobs, &screen_descriptor)
                .unwrap();
        }

        self.queue.submit(Some(encoder.finish()));

        frame.present();
    }

    pub fn update(&mut self) {
        let offset = 100 * 4;
        for i in 0..100 {
            for j in 0..4 {
                self.texels[offset + i * 4 + j] += 1;
            }
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop: EventLoop<()> = EventLoop::with_user_event();
    let window = WindowBuilder::new()
        .with_title("EMM")
        .with_inner_size(winit::dpi::PhysicalSize {
            width: DISPLAY_WIDTH * 4,
            height: DISPLAY_HEIGHT * 4,
        })
        .with_min_inner_size(winit::dpi::PhysicalSize {
            width: DISPLAY_WIDTH,
            height: DISPLAY_HEIGHT,
        })
        .build(&event_loop)
        .unwrap();

    let mut app = pollster::block_on(App::new(&window));

    event_loop.run(move |event, _, control_flow| {
        app.egui_context.platform.handle_event(&event);
        match event {
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
                //app.update();
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                app.render(&window);
            }
            _ => (),
        }
    });
}
