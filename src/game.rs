use crate::consts::*;

pub struct Game {
    // simple for now
    texels: Vec<u8>,
}

impl Game {
    pub fn new() -> Self {
        Self {
            texels: create_texels(),
        }
    }

    pub fn update(&mut self) {
        let offset = 0;
        for i in 0..10 {
            for j in 0..4 {
                if self.texels[offset + i * 4 + j] == u8::MAX {
                    self.texels[offset + i * 4 + j] = 0;
                } else {
                    self.texels[offset + i * 4 + j] += 1;
                }
            }
        }
    }

    pub fn texels(&self) -> &[u8] {
        &self.texels
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
