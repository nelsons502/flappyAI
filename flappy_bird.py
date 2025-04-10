import pygame
import sys
import random
import socket


# Initialize Pygame
pygame.init()

background = pygame.image.load("images/background.png")

# Constants
WIDTH, HEIGHT = background.get_size()
FPS = 60
GRAVITY = 0.25
FLAP_PWR = -6 
PIPE_GAP = 150
score = 0
game_over = False
pipe_timer = 0
mode = "play"

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()

# Load images
flat = pygame.image.load("images/bird0.png")
up = pygame.image.load("images/bird1.png")
down = pygame.image.load("images/bird2.png")
dead = pygame.image.load("images/birddead.png")
fly_images = [up, flat, down, flat]

top_img = pygame.image.load("images/top.png")
bottom_img = pygame.image.load("images/bottom.png")


# Bird class as an Actor/Sprite
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = flat
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 4, HEIGHT // 2)
        self.speed = 0
    
    def update(self):
        global game_over
        self.speed += GRAVITY
        self.rect.centery += self.speed
        self.image = fly_images[int(self.speed // 10) % 4] if not game_over else dead
        # change angle of bird based on speed
        self.image = pygame.transform.rotozoom(self.image, -self.speed * 3, 1)
    
    def flap(self):
        self.speed = FLAP_PWR

# Pipe class as an Actor/Sprite
class Pipe(pygame.sprite.Sprite):
    def __init__(self, image, pos):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect()
        if image == top_img:
            self.rect.midbottom = pos
        else:
            pos = (WIDTH, pos[1] + PIPE_GAP)
            self.rect.midtop = pos
        self.speed = 3
        self.scored = False
    
    def update(self):
        self.rect.centerx -= self.speed
        if self.rect.right < 0: # remove pipe from the game
            self.kill()
        
    def collide(self, bird):
        return self.rect.colliderect(bird.rect)
    
    def score(self, bird):
        if self.rect.right < bird.rect.left:
            if not self.scored:
                self.scored = True
                return True
        return False


def connect_to_ai():
    host = "localhost"
    port = 12345
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        return client_socket
    except Exception as e:
        print("Error connecting to AI server:", e)
        sys.exit(1)

def send_state_and_receive_action(client_socket, state):
    try:
        state_str = ','.join(map(str, state))
        client_socket.sendall(state_str.encode('utf-8'))
        data = client_socket.recv(1024).decode('utf-8')
        action = int(data.strip())
        return action
    except Exception as e:
        print("Error in sending state or receiving action:", e)
        return 0

client_socket = connect_to_ai()

def __main__():
    global score, game_over, pipe_timer
        
    all_sprites = pygame.sprite.Group()
    pipes = pygame.sprite.Group()

    flappy = Bird()
    all_sprites.add(flappy)

    def spawn_pipes():
        pipe_pos = random.randint(50, HEIGHT - 200)
        top = Pipe(top_img, (WIDTH, pipe_pos))
        bottom = Pipe(bottom_img, (WIDTH, pipe_pos))
        all_sprites.add(top, bottom)
        pipes.add(top, bottom)

    def draw_score():
        global score
        font = pygame.font.Font(None, 50 )
        text = font.render(str(score//2), True, (0,0,0))
        screen.blit(text, (WIDTH // 2, 50))

    def reset():
        global score, game_over, pipe_timer
        score = 0
        game_over = False
        pipe_timer = 0
        flappy.rect.center = (WIDTH // 4, HEIGHT // 2)
        flappy.speed = 0
        all_sprites.empty()
        pipes.empty()
        all_sprites.add(flappy)
        flappy.image = flat
    
    def get_game_state():
        MAX_PIPES = 4
        state_vect = [flappy.rect.centery / HEIGHT, flappy.speed / 10]

        for pipe in pipes:
            state_vect.append(pipe.rect.centerx / WIDTH)
            state_vect.append(pipe.rect.centery / HEIGHT)
            state_vect.append(pipe.speed / 10)

        # Pad with zeros if fewer pipes
        while len(state_vect) < 2 + MAX_PIPES * 3:
            state_vect.append(0.0)
        
        return state_vect

        ''' # Alternative state vector with compact info about the whole state
        if pipes:
            nearest_pipe = min(pipes, key=lambda p: p.rect.centerx - flappy.rect.centerx)
            state_vect.extend([
                nearest_pipe.rect.centerx / WIDTH,
                nearest_pipe.rect.centery / HEIGHT,
                nearest_pipe.speed / 10,
            ])
        else:
            state_vect.extend([0.0, 0.0, 0.0])
            '''

    while True:
        screen.blit(background, (0, 0))
        draw_score()

        if mode == "train":
            state = get_game_state()
            #print(state)
            action = send_state_and_receive_action(client_socket, state)
            if action == 1 and not game_over:
                flappy.flap()
        else:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if not game_over and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        flappy.flap()
        
        all_sprites.update()
        
        pipe_timer += 1
        if pipe_timer == 100 and not game_over:
            pipe_timer = 0
            spawn_pipes()
        
        for pipe in pipes:
            if not game_over and pipe.collide(flappy) or flappy.rect.top < 0 or flappy.rect.bottom > HEIGHT:
                flappy.image = dead
                flappy.speed = 0
                pipes.empty()
                game_over = True
                pipe_timer = 0
                break
            if not game_over and pipe.score(flappy):
                score += 1
        
        all_sprites.draw(screen)

        if game_over:
            font = pygame.font.Font(None, 50)
            text = font.render("Game Over", True, (200,0,0))
            screen.blit(text, (WIDTH // 4, HEIGHT // 2))
            if pipe_timer == 100:
                reset()
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    # set mode to train if train argument included
    mode = "train" if "train" in sys.argv else "play"
    __main__()
    
# flappy_bird.py