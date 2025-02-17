import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 576, 1024
FPS = 60
GRAVITY = 0.25
FLAP_PWR = -5

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()

# Load images
flat = pygame.image.load("images/bird0.png")
up = pygame.image.load("images/bird1.png")
down = pygame.image.load("images/bird2.png")
dead = pygame.image.load("images/birddead.png")
fly_images = [up, flat, down, flat]

top_pipe = pygame.image.load("images/top.png")
bottom_pipe = pygame.image.load("images/bottom.png")

background = pygame.image.load("images/background.png")


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
        self.image = fly_images[(self.speed // 10) % 4]
    
    def flap(self):
        self.speed = FLAP_PWR

# Pipe class as an Actor/Sprite
class Pipe(pygame.sprite.Sprite):
    def __init__(self, image, pos):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.midbottom = pos
        self.speed = 3
    
    def update(self):
        self.rect.centerx -= self.speed
        if self.rect.right < 0:
            self.kill()
        
    def collide(self, bird):
        return self.rect.colliderect(bird.rect)
    
