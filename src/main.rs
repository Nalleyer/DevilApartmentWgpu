mod consts;
mod game;
mod render;
mod ui;
mod ui_context;

use consts::*;
use game::Game;
use render::Renderer;
use ui_context::EguiContext;

use winit::dpi::PhysicalSize;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

struct App {
    game: Game,
    renderer: Renderer,
    egui_context: EguiContext,
}

impl App {
    pub async fn new(window: &Window) -> Self {
        let renderer = Renderer::new(window).await;
        let game = Game::new();
        /*
        queue.write_texture(
            texture.as_image_copy(),
            &game.texels(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(DISPLAY_WIDTH * 4u32).unwrap()),
                rows_per_image: None,
            },
            texture_extent,
        );
        */

        let egui_context = EguiContext::new(window, renderer.device(), renderer.texture_format());

        Self {
            game,
            renderer,
            egui_context,
        }
    }

    pub fn resize(&mut self, size: &PhysicalSize<u32>) {
        self.renderer.resize(size);
    }

    pub fn render(&mut self, window: &Window) {
        self.renderer
            .render(window, self.game.texels(), &mut self.egui_context);
    }

    pub fn update(&mut self) {
        self.game.update();
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
        app.egui_context.handle_event(&event);
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
                app.update();
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                app.render(&window);
            }
            _ => (),
        }
    });
}
