import pygame
import sys
import random

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
        self.speed += GRAVITY
        self.rect.centery += self.speed
        self.image = fly_images[int(self.speed // 10) % 4]
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
    font = pygame.font.Font(None, 50 )
    text = font.render(str(score//2), True, (0,0,0))
    screen.blit(text, (WIDTH // 2, 50))

game_over = False
pipe_timer = 0

while True:
    screen.blit(background, (0, 0))
    draw_score()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
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
        font = pygame.font.Font(None, 100)
        text = font.render("Game Over", True, (200,0,0))
        screen.blit(text, (WIDTH // 4, HEIGHT // 2))
        if pipe_timer == 100:
            break

    
    
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
    

