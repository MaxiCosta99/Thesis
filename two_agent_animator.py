import pygame
import sys
import numpy as np
import os



def save_frame(screen, frame_idx, out_dir="frames"):
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/frame_{frame_idx:05d}.png"
    pygame.image.save(screen, filename)


def lerp(a, b, t):
    return a + t * (b - a)

def grid_origin(grid_idx, ROWS, CELL_SIZE, GRID_MARGIN):
    y = grid_idx * (ROWS * CELL_SIZE + GRID_MARGIN)
    return 0, y

def cell_center(origin, row, col, CELL_SIZE):
    ox, oy = origin
    x = ox + col * CELL_SIZE + CELL_SIZE // 2
    y = oy + row * CELL_SIZE + CELL_SIZE // 2
    return [x, y]


class EnvState:
    def __init__(self, positions, cheeses, agent_worlds):
        # One agent per grid
        self.agent_pos = [(0, 1), (0, 1)]
        self.turn = 0  # which agent acts
        self.positions = positions
        self.cheese_pos = []
        for i in range(2):
            if cheeses[i] == 0:
                self.cheese_pos.append((0, 0))
            else:
                self.cheese_pos.append((0, 2))
        self.agent_worlds = agent_worlds


    def step(self):
        """
        One turn update.
        Placeholder deterministic movement:
        right → down → left → up
        """
        i = self.turn
        
        self.agent_pos[i] = self.positions.pop(0)[i]

        self.turn = 1 - self.turn


class Renderer:
    def __init__(self, screen, env, cheese_pos, COLS, ROWS, CELL_SIZE, GRID_MARGIN):
        self.screen = screen
        self.env = env
        self.cheese_pos = cheese_pos
        self.COLS = COLS
        self.ROWS = ROWS
        self.CELL_SIZE = CELL_SIZE
        self.GRID_MARGIN = GRID_MARGIN
        self.CHEESE_COLOR = (200, 200, 80)
        self.clue_pos = (1,1)
        self.CLUE_COLOR = (80, 200, 80)

        self.render_pos = []
        self.target_pos = []
        self.clue_pos = (1,1)

        self.agent0_world = env.agent_worlds[0]
        self.agent1_world = env.agent_worlds[1]

        for i in range(2):
            origin = grid_origin(i,ROWS, CELL_SIZE, GRID_MARGIN)
            p = cell_center(origin, *env.agent_pos[i], CELL_SIZE)
            self.render_pos.append(p.copy())
            self.target_pos.append(p.copy())

    def sync_targets(self):
        for i in range(2):
            origin = grid_origin(i, self.ROWS, self.CELL_SIZE, self.GRID_MARGIN)
            self.target_pos[i] = cell_center(
                origin, *self.env.agent_pos[i], self.CELL_SIZE
            )


    def is_animating(self):
        for i in range(2):
            dx = abs(self.render_pos[i][0] - self.target_pos[i][0])
            dy = abs(self.render_pos[i][1] - self.target_pos[i][1])
            if dx > 1 or dy > 1:
                return True
        return False
    

    
    def is_animating_strict(self):
        for i in range(2):
            dx = abs(self.render_pos[i][0] - self.target_pos[i][0])
            dy = abs(self.render_pos[i][1] - self.target_pos[i][1])
            if dx > 0 or dy > 0:
                return True
        return False


    def update(self, LERP_ALPHA):
        for i in range(2):
            for k in (0, 1):
                self.render_pos[i][k] = lerp(
                    self.render_pos[i][k],
                    self.target_pos[i][k],
                    LERP_ALPHA
                )

    def draw_grid(self, origin, ROWS, COLS, CELL_SIZE, GRID_COLOR, ACTIVE_BORDER, active=False):
        ox, oy = origin

        # Horizontal lines
        for r in range(ROWS + 1):
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (ox, oy + r * CELL_SIZE),
                (ox + COLS * CELL_SIZE, oy + r * CELL_SIZE)
            )

        # Vertical lines
        for c in range(COLS + 1):
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (ox + c * CELL_SIZE, oy),
                (ox + c * CELL_SIZE, oy + ROWS * CELL_SIZE)
            )

        if active:
            pygame.draw.rect(
                self.screen,
                ACTIVE_BORDER,
                pygame.Rect(
                    ox - 4, oy - 4,
                    COLS * CELL_SIZE + 8,
                    ROWS * CELL_SIZE + 8
                ),
                width=3
            )

    def draw_agents(self, AGENT_COLORS, CELL_SIZE):
        if self.agent0_world == 0:
            pygame.draw.circle(
                self.screen,
                AGENT_COLORS[0],
                self.render_pos[0],
                CELL_SIZE // 5
            )
        else:
            pygame.draw.circle(
                self.screen,
                AGENT_COLORS[0],
                self.render_pos[1],
                CELL_SIZE // 5
            )

        if self.agent1_world == 0:
            pygame.draw.circle(
                self.screen,
                AGENT_COLORS[1],
                self.render_pos[0],
                CELL_SIZE // 5
            )
        else:
            pygame.draw.circle(
                self.screen,
                AGENT_COLORS[1],
                self.render_pos[1],
                CELL_SIZE // 5
            )

    def draw_cheese(self):
        for i in range(2):
            origin = grid_origin(i, self.ROWS, self.CELL_SIZE, self.GRID_MARGIN)
            pygame.draw.circle(
                self.screen,
                self.CHEESE_COLOR,
                cell_center(origin,*self.cheese_pos[i], self.CELL_SIZE),
                self.CELL_SIZE // 3
            )

    def draw_clue(self):
        for i in range(2):
            origin = grid_origin(i, self.ROWS, self.CELL_SIZE, self.GRID_MARGIN)
            pygame.draw.circle(
                self.screen,
                self.CLUE_COLOR,
                cell_center(origin,*self.clue_pos, self.CELL_SIZE),
                self.CELL_SIZE // 3
            )

    def render(self, BG_COLOR, GRID_COLOR, AGENT_COLORS, CELL_SIZE, ACTIVE_BORDER, COLS, ROWS, GRID_MARGIN):
        self.screen.fill(BG_COLOR)

        for i in range(2):
            self.draw_grid(
                grid_origin(i, ROWS, CELL_SIZE, GRID_MARGIN),
                ROWS, COLS, CELL_SIZE, GRID_COLOR, ACTIVE_BORDER,
                active=(self.env.turn == i)
            )
        self.draw_cheese()
        self.draw_clue()
        self.draw_agents(AGENT_COLORS, CELL_SIZE)
        
        #pygame.display.flip()



def prob_to_gray(p):
    #assert p >= 0 and p <= 1, "Probability must be in [0,1]"
    p = max(0.0, min(1.0, p))
    v = int(255 * p)
    return (v, v, v)


class Matrix_Renderer:
    def __init__(
        self,
        screen,
        top_left,
        cell_size=10,
        grid_color=(80, 80, 80),
        os_matrix = None,
        reduction = str
    ):
        self.screen = screen
        self.top_left = top_left
        self.cell_size = cell_size
        self.grid_color = grid_color
        if reduction is not None:
            self.os_matrix = np.einsum(reduction, os_matrix)
            self.os_matrix = self.os_matrix.reshape(self.os_matrix.shape[0], -1)  # Flatten last dimensions
            self.os_matrix = self.os_matrix/np.sum(self.os_matrix, axis = 0, keepdims = True)
            assert np.allclose(np.sum(self.os_matrix, axis = 0), 1), "Columns of Alpha_matrix must sum to 1" 

        else:
            self.os_matrix = os_matrix

        #print(f"This is the shape of the A matrix: {self.os_matrix.shape}")

        

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

def animate(epoch_positions, cheeses, A_matrices, agent_worlds, epochs_beliefs, epochs_desires):
    

    CELL_SIZE = 100
    ROWS = 2
    COLS = 3
    GRID_MARGIN = 100
    

    FPS = 60
    LERP_ALPHA = 0.2

    WINDOW_WIDTH = 2 * COLS * CELL_SIZE + GRID_MARGIN + 500
    WINDOW_HEIGHT = ROWS * CELL_SIZE + 500# + 500 #EXTRA SPACE FOR MATRICES

    ww = COLS * CELL_SIZE + GRID_MARGIN
    wh = 2 * ROWS * CELL_SIZE + GRID_MARGIN

    BG_COLOR = (25, 25, 25)
    GRID_COLOR = (90, 90, 90)
    AGENT_COLORS = [(220, 80, 80), (80, 160, 220)]
    ACTIVE_BORDER = (200, 200, 80)


    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )
    pygame.display.set_caption("Two-Agent, Two 2x3 Grids")
    clock = pygame.time.Clock()

    step_idx = 0

    for i in range(len(epoch_positions)):

        beliefs = epochs_beliefs[i]
        desires = epochs_desires[i]

        env = EnvState(epoch_positions[i], cheeses[i], agent_worlds[i])
        world_renderer = Renderer(screen, env, env.cheese_pos, COLS, ROWS, CELL_SIZE, GRID_MARGIN)
        
        matrix_pos = [[ww, 0], [ww+100, 0],[ww+300, 0],[ww+400, 0]]
        reductions = ["oijkl -> oik", "oijkl -> ojl", "oijkl -> ojl",  "oijkl -> oik"]
        matrices = [A_matrices[i][0][0],
                    A_matrices[i][0][1],
                    A_matrices[i][1][0],
                    A_matrices[i][1][1]]

        matrix_renderers = []
        for mp, r, m in zip(matrix_pos, reductions, matrices):
            matrix_renderers.append(Matrix_Renderer(screen, top_left = mp, os_matrix = m, reduction = r))

        curr_bel = beliefs.pop(0)

        matrices = []
        matrix_pos = []
        for m in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        matrices.append(curr_bel[m][j,k,l,:])
                        matrix_pos.append([ww+100*l+200*j+400*m,200+100*k])
        
        bm_renderers = []
        for mp, m in zip(matrix_pos, matrices):
            bm_renderers.append(Matrix_Renderer(screen, top_left = mp, os_matrix = m, reduction = None))


        mh = 200
        des_matrices = []
        des_matrix_pos = []
        for m in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        des_matrices.append(desires[m][j,k,l,:]*10)
                        des_matrix_pos.append([ww+100*l+200*j+400*m,mh+200+100*k])
                
        dm_renderers = []
        for mp, m in zip(des_matrix_pos, des_matrices):
            dm_renderers.append(Matrix_Renderer(screen, top_left = mp, os_matrix = m, reduction = None))

    
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
                        if not world_renderer.is_animating():
                            try:
                                env.step()
                                world_renderer.sync_targets()
                                curr_bel = beliefs.pop(0)

                                
                                matrices = []
                                matrix_pos = []
                                for i in range(2):
                                    for j in range(2):
                                        for k in range(2):
                                            for l in range(2):
                                                matrices.append(curr_bel[i][j,k,l,:])
                                                matrix_pos.append([ww+100*l+200*j+400*i,200+100*k])
                                
                                bm_renderers = []
                                for mp, m in zip(matrix_pos, matrices):
                                    bm_renderers.append(Matrix_Renderer(screen, top_left = mp, os_matrix = m, reduction = None))

                    
                            except:
                                running = False
                            

                    if event.key == pygame.K_p:
                        #save_frame(screen, step_idx)
                        #step_idx += 1
                        print(f"I just saved frame {step_idx}")

            


            world_renderer.update(LERP_ALPHA)
            world_renderer.render(BG_COLOR, GRID_COLOR, AGENT_COLORS, CELL_SIZE, ACTIVE_BORDER, COLS, ROWS, GRID_MARGIN)
            
            for mr in matrix_renderers:
                mr.render()
            

            for bmr in bm_renderers:
                bmr.render()

            for dmr in dm_renderers:
                dmr.render()

            pygame.display.flip()


    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    animate()
