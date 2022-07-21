use egui::FontDefinitions;
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::{Platform, PlatformDescriptor};
use winit::window::Window;

pub struct EguiContext {
    render_pass: RenderPass,
    platform: Platform,
}

impl EguiContext {
    pub fn new(
        window: &Window,
        device: &wgpu::Device,
        surface_texture_format: wgpu::TextureFormat,
    ) -> Self {
        let platform = Platform::new(PlatformDescriptor {
            physical_width: window.inner_size().width,
            physical_height: window.inner_size().height,
            scale_factor: window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: Default::default(),
        });

        let render_pass = RenderPass::new(device, surface_texture_format, 1);

        Self {
            render_pass,
            platform,
        }
    }

    pub fn render<'r, 's: 'r>(
        &'s mut self,
        window: &Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        mut render_pass: wgpu::RenderPass<'r>,
    ) {
        let screen_descriptor = ScreenDescriptor {
            physical_width: surface_config.width,
            physical_height: surface_config.height,
            scale_factor: window.scale_factor() as f32,
        };

        self.platform.begin_frame();

        // all ui logic here
        egui::Window::new("fps").show(&self.platform.context(), |ui| {
            ui.label("fps: unknown");
        });

        let full_output = self.platform.end_frame(Some(window));
        let paint_jobs = self.platform.context().tessellate(full_output.shapes);

        let tdelta: egui::TexturesDelta = full_output.textures_delta;
        self.render_pass
            .add_textures(device, queue, &tdelta)
            .expect("add texture ok");

        self.render_pass
            .update_buffers(device, queue, &paint_jobs, &screen_descriptor);

        self.render_pass
            .execute_with_renderpass(&mut render_pass, &paint_jobs, &screen_descriptor)
            .unwrap();
    }

    pub fn handle_event(&mut self, event: &winit::event::Event<()>) {
        self.platform.handle_event(event);
    }
}
