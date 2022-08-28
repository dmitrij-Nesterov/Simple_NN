import pygame
import numpy as np
import random
pygame.init()

def relu(x, deriv=False):
	if deriv:
		return (x >= 0).astype(float)
	return np.maximum(x, 0)
def softmax(x):
	out = np.exp(x)
	return out / np.sum(out)
def to_full(y, num_classes):
	y_full = np.zeros((1, num_classes))
	y_full[0, y] = 1
	return y_full
def get_accuracy():
	correct = 0
	for x, y in dataset:
		t1 = x @ synaptic_weights_input + b_input
		h1 = relu(t1)
		t2 = h1 @ synaptic_weights_hidden1 + b_hidden1
		h2 = relu(t2)
		t3 = h2 @ synaptic_weights_hidden2 + b_hidden2
		h3 = relu(t3)
		t4 = h3 @ synaptic_weights_hidden3 + b_hidden3
		z = softmax(t4)
		pred_y = np.argmax(z)
		if pred_y == y:
			correct += 1
	return correct / len(dataset) * 100

FPS = 60
WIDTH = 625
HEIGHT = 625
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
ALPHA = 0.00094
CELL_LEN = 25
NUM_INPUTS = 625
NUM_HIDDEN = 125
NUM_OUTPUTS = 2
NUM_EPOCH = 10000
clock = pygame.time.Clock()
dataset = [(np.zeros((1, NUM_INPUTS)), )]
n = 0
classes = ['Грустный:(', 'Веселый:)']

synaptic_weights_input = np.random.randn(NUM_INPUTS, NUM_HIDDEN)
b_input = np.random.randn(1, NUM_HIDDEN)
synaptic_weights_hidden = np.random.randn(NUM_HIDDEN, NUM_OUTPUTS)
b_hidden = np.random.randn(1, NUM_OUTPUTS)

synaptic_weights_input = (synaptic_weights_input-0.5)*2*np.sqrt(1/NUM_INPUTS)
b_input = (b_input-0.5)*2*np.sqrt(1/NUM_INPUTS)
synaptic_weights_hidden = (synaptic_weights_hidden-0.5)*2*np.sqrt(1/NUM_HIDDEN)
b_hidden = (b_hidden-0.5)*2*np.sqrt(1/NUM_HIDDEN)

font = pygame.font.SysFont('arial', 36)
cord_text = (150, 90)
text = font.render('', True, WHITE)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Нейронка')

run = True
while run:
	clock.tick(FPS)
	mouse_pressed = pygame.mouse.get_pressed()
	pos = pygame.mouse.get_pos()

	if mouse_pressed[0]:
		dataset[n][0][0][pos[1]-pos[1]%int(np.sqrt(NUM_INPUTS))+pos[0]//CELL_LEN] = 1
	elif mouse_pressed[2]:
		dataset[n][0][0][pos[1]-pos[1]%int(np.sqrt(NUM_INPUTS))+pos[0]//CELL_LEN] = 0

	for i in range(len(dataset[n][0][0])):
		if dataset[n][0][0][i]:
			pygame.draw.rect(screen, BLUE, [i%int(np.sqrt(NUM_INPUTS))*int(np.sqrt(NUM_INPUTS)), i-i%int(np.sqrt(NUM_INPUTS)), CELL_LEN, CELL_LEN])
	for i in range(0, WIDTH, CELL_LEN):
		pygame.draw.line(screen, WHITE, [i, 0], [i, HEIGHT])
		pygame.draw.line(screen, WHITE, [0, i], [WIDTH, i])

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
		if event.type == pygame.KEYUP:
			if event.key == pygame.K_c:
				cord_text = (150, 90)
				dataset[n][0][0] = np.zeros((1, NUM_INPUTS))
				text = font.render('', True, WHITE)
			if event.key == pygame.K_p:
				cord_text = (150, 90)
				dataset.append((np.zeros((1, NUM_INPUTS)),))
				dataset[n] = dataset[n] + (1,)
				n += 1
				text = font.render('', True, WHITE)
			if event.key == pygame.K_n:
				cord_text = (150, 90)
				dataset.append((np.zeros((1, NUM_INPUTS)),))
				dataset[n] = dataset[n] + (0,)
				n += 1
				text = font.render('', True, WHITE)
			if event.key == pygame.K_q:
				cord_text = (150, 90)
				dataset = [(np.zeros((1, NUM_INPUTS)),)]
				n = 0
			if event.key == pygame.K_s:
				if len(dataset) == 1:
					cord_text = (10, 90)
					text = font.render('А сообствено, на каких данных мне учиться?', True, WHITE)
				else:
					n -= 1
					dataset.pop(-1)
					for epoch in range(NUM_EPOCH):
						random.shuffle(dataset)
						for i in range(0, WIDTH, CELL_LEN):
							pygame.draw.line(screen, WHITE, [i, 0], [i, HEIGHT])
							pygame.draw.line(screen, WHITE, [0, i], [WIDTH, i])
						screen.blit(font.render(f"Эпоха[{epoch+1}/{NUM_EPOCH}]", True, WHITE), (150, 90))
						pygame.display.update()
						screen.fill((0, 0, 0))

						for i in range(len(dataset)):
							x, y = dataset[i]
							t1 = x @ synaptic_weights_input + b_input
							h1 = relu(t1)
							t2 = h1 @ synaptic_weights_hidden + b_hidden
							z = softmax(t2)

							y_full = to_full(y, NUM_OUTPUTS)
							dE_dt2 = z - y_full
							dE_dW2 = h1.T @ dE_dt2
							dE_db2 = dE_dt2
							dE_dh1 = dE_dt2 @ synaptic_weights_hidden.T
							dE_dt1 = dE_dh1 * relu(t1, True)
							dE_db1 = dE_dt1
							dE_W1 = x.T @ dE_dt1

							synaptic_weights_input -= ALPHA * dE_W1
							b_input -= ALPHA * dE_db1
							synaptic_weights_hidden -= ALPHA * dE_dW2
							b_hidden -= ALPHA * dE_db2

				cord_text = (25, 90)
				text = font.render(f'Кол-во верных ответов: {get_accuracy()}%', True, WHITE)

			if event.key == pygame.K_r:
				cord_text = (150, 90)
				x = dataset[n][0][0]
				t1 = x @ synaptic_weights_input + b_input
				h1 = relu(t1)
				t2 = h1 @ synaptic_weights_hidden1 + b_hidden1
				h2 = relu(t2)
				t3 = h2 @ synaptic_weights_hidden2 + b_hidden2
				h3 = relu(t3)
				t4 = h3 @ synaptic_weights_hidden3 + b_hidden3
				z = softmax(t4)
				text = font.render(classes[np.argmax(z)], True, WHITE)

	screen.blit(text, cord_text)
	pygame.display.update()
	screen.fill((0, 0, 0))