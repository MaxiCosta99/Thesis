
from py_compile import main
import pygame
import sys
import numpy as np



def lerp(a, b, t):
    return a + t * (b - a)

def cell_center(row, col, CELL_SIZE):
    x = col * CELL_SIZE + CELL_SIZE // 2
    y = row * CELL_SIZE + CELL_SIZE // 2
    return [x, y]

def prob_to_gray(p):
    assert p >= 0 and p <= 1, "Probability must be in [0,1]"
    p = max(0.0, min(1.0, p))
    v = int(255 * p)
    return (v, v, v)


class Matrix_Renderer:
    def __init__(
        self,
        screen,
        top_left,
        cell_size=30,
        grid_color=(80, 80, 80),
        os_matrix = None
    ):
        self.screen = screen
        self.top_left = top_left
        self.cell_size = cell_size
        self.grid_color = grid_color
        self.os_matrix = os_matrix

        self.os_matrix = os_matrix/np.sum(os_matrix, axis = 0, keepdims = True)

        assert np.allclose(np.sum(self.os_matrix, axis = 0), 1), "Columns of Alpha_matrix must sum to 1" 

    def render(self):
        rows = len(self.os_matrix)        # |O|
        cols = len(self.os_matrix[0])     # |S|
        ox, oy = self.top_left

        # Cells
        for o in range(rows):
            for s in range(cols):
                p = self.os_matrix[o][s]
                color = prob_to_gray(p)

                rect = pygame.Rect(
                    ox + s * self.cell_size,
                    oy + o * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

        # Grid lines
        for o in range(rows + 1):
            pygame.draw.line(
                self.screen, self.grid_color,
                (ox, oy + o * self.cell_size),
                (ox + cols * self.cell_size,
                 oy + o * self.cell_size)
            )

        for s in range(cols + 1):
            pygame.draw.line(
                self.screen, self.grid_color,
                (ox + s * self.cell_size, oy),
                (ox + s * self.cell_size,
                 oy + rows * self.cell_size)
            )

    def update_matrix(self, obs_model):
        self.os_matrix = obs_model


class EnvState:
    def __init__(self, pos, cheese_pos):
        # Discrete grid position (row, col)
        #self.agent_pos = (0, 0)
        self.agent_pos = (0,1)
        self.pos = pos
        if cheese_pos == 0:
            self.cheese_pos = (0,0)
        else:
            self.cheese_pos = (0,2)
        self.clue_pos = (1,1)

    def step(self):
        self.agent_pos = self.pos.pop(0)



class Renderer:
    def __init__(self, screen, env, cheese_pos, CELL_SIZE, ROWS,
                 COLS, GRID_COLOR, ACTIVE_BORDER,
                 AGENT_COLOR, BG_COLOR, LERP_ALPHA):
        self.screen = screen
        self.env = env
        self.cheese_pos = cheese_pos
        self.CELL_SIZE = CELL_SIZE
        self.ROWS = ROWS
        self.COLS = COLS
        self.GRID_COLOR = GRID_COLOR
        self.ACTIVE_BORDER = ACTIVE_BORDER
        self.AGENT_COLOR = AGENT_COLOR
        self.BG_COLOR = BG_COLOR
        self.LERP_ALPHA = LERP_ALPHA
        self.CHEESE_COLOR = (200, 200, 80)
        self.clue_pos = (1,1)
        self.clue_color = (80, 200, 80),




        # Continuous render position
        self.render_pos = cell_center(*env.agent_pos, CELL_SIZE)
        self.target_pos = self.render_pos.copy()

    def sync_target(self):
        """Update target position after env step."""
        self.target_pos = cell_center(*self.env.agent_pos, self.CELL_SIZE)

    def is_animating(self):
        dx = abs(self.render_pos[0] - self.target_pos[0])
        dy = abs(self.render_pos[1] - self.target_pos[1])
        return dx > 1 or dy > 1

    def update(self):
        """Smooth interpolation every frame."""
        for k in (0, 1):
            self.render_pos[k] = lerp(
                self.render_pos[k],
                self.target_pos[k],
                self.LERP_ALPHA
            )

    def draw_grid(self):
        # Horizontal lines
        for r in range(self.ROWS + 1):
            pygame.draw.line(
                self.screen, self.GRID_COLOR,
                (0, r * self.CELL_SIZE),
                (self.COLS * self.CELL_SIZE, r * self.CELL_SIZE)
            )

        # Vertical lines
        for c in range(self.COLS + 1):
            pygame.draw.line(
                self.screen, self.GRID_COLOR,
                (c * self.CELL_SIZE, 0),
                (c * self.CELL_SIZE, self.ROWS * self.CELL_SIZE)
            )

        # Border
        pygame.draw.rect(
            self.screen,
            self.ACTIVE_BORDER,
            pygame.Rect(
                -4, -4,
                self.COLS * self.CELL_SIZE + 8,
                self.ROWS * self.CELL_SIZE + 8
            ),
            width=3
        )

    def draw_agent(self):
        pygame.draw.circle(
            self.screen,
            self.AGENT_COLOR,
            self.render_pos,
            self.CELL_SIZE // 5
        )
    
    def draw_cheese(self):
        pygame.draw.circle(
            self.screen,
            self.CHEESE_COLOR,
            cell_center(*self.cheese_pos, self.CELL_SIZE),
            self.CELL_SIZE // 3
        )

    def draw_clue(self):
        pygame.draw.circle(
            self.screen,
            self.clue_color,
            cell_center(*self.clue_pos, self.CELL_SIZE),
            self.CELL_SIZE // 3
        )

    def render(self):
        self.screen.fill(self.BG_COLOR)
        self.draw_grid()
        self.draw_cheese()
        self.draw_clue()
        self.draw_agent()
        #pygame.display.flip()



def animate(states, cheeses, A_matrices):
    CELL_SIZE = 100
    ROWS = 2        # vertical
    COLS = 3        # horizontal
    FPS = 60
    LERP_ALPHA = 0.9

    WINDOW_WIDTH = COLS * CELL_SIZE + 210
    WINDOW_HEIGHT = ROWS * CELL_SIZE

    BG_COLOR = (25, 25, 25)
    GRID_COLOR = (90, 90, 90)
    AGENT_COLOR = (220, 80, 80)
    ACTIVE_BORDER = (200, 200, 80)

    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )
    pygame.display.set_caption("Single-Agent 2x3 Grid")
    clock = pygame.time.Clock()



 


    
    for i, state in enumerate(states):
        #print(f"This is the matrix A = {np.round(A_matrices[i],1)}")
        env = EnvState(state, cheese_pos = cheeses[i])
        world_renderer = Renderer(screen, env, env.cheese_pos, CELL_SIZE, ROWS, COLS,
                            GRID_COLOR, ACTIVE_BORDER, AGENT_COLOR,
                            BG_COLOR, LERP_ALPHA)
        
        matrix_renderer = Matrix_Renderer(
            screen=screen,
            top_left=(310, 0),  # YOU control placement
            cell_size=25,
            os_matrix = A_matrices[i]
)
        

        running = True
        while running:
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    if event.key == pygame.K_SPACE:
                        # Enforce: one action per completed animation
                        if not world_renderer.is_animating():
                            try:
                                env.step()
                                world_renderer.sync_target()
                            except:
                                running = False


            world_renderer.update()
            world_renderer.render()
            matrix_renderer.render()
            pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    animate()
