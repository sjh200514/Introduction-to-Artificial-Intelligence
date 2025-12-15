import pygame
import sys
from pygame.locals import *
import numpy as np
from enum import Enum

pygame.init()
pygame.display.set_caption('SJH的五子棋')
screen = pygame.display.set_mode((800,900))
font = pygame.font.Font(None, 100)
screen_color=[255,165,79] #设置背景为棕黄色
line_color = [0,0,0] #设置棋盘的线为黑色
rect_color = [0,239,248] #设置落子悬浮区域颜色
white_chess_color = [255,255,255] #白棋的颜色
black_chess_color = [0,0,0] #黑棋的颜色
exist_chess = []
position = np.zeros((15,15),dtype=int)
record = [[[0,0,0,0]for _ in range(15)] for _ in range(15)]
count = [[0 for _ in range(8)] for _ in range(2)]
tim = 0
turn = 1
flag = False
game_start = False
board = np.zeros((15, 15), dtype=int)
directions = [(0, 1),(1, 0),(1, 1),(1, -1),(0, -1),(-1, 0),(-1, 1),(-1, -1)]
direction = [(1, 0), (0, 1), (1, 1), (1, -1)]

button_black = pygame.Rect(50,820,200,70)
button_white = pygame.Rect(550,820,200,70)
button_restart = pygame.Rect(300,820,150,70)
black_color = (255, 255, 255)
text_color = (0, 0, 0)
button_font = pygame.font.SysFont(['方正粗黑宋简体','microsoftsansserif'],30)
button_text = button_font.render("Choose Black", True, text_color)
button_text2 = button_font.render("Choose White", True, text_color)
button_text3 = button_font.render("Restart", True, text_color)

win_text = font.render(None, True, (255, 0, 0))
text_rect = win_text.get_rect()


ai_num = 0
player_num = 0



for i in range(15):
  for j in  range(15):
    position[i][j] = 7-max(abs(i-7),abs(j-7))

#绘制棋盘
def draw_map():
    screen.fill(screen_color)
    pygame.draw.rect(screen, black_color, button_black)
    screen.blit(button_text, (button_black.centerx - button_text.get_width() / 2, button_black.centery - button_text.get_height() / 2))
    pygame.draw.rect(screen, black_color, button_white)
    screen.blit(button_text2, (button_white.centerx - button_text2.get_width() / 2, button_white.centery - button_text2.get_height() / 2))
    pygame.draw.rect(screen, black_color, button_restart)
    screen.blit(button_text3, (button_restart.centerx - button_text3.get_width() / 2, button_restart.centery - button_text3.get_height() / 2))
    if not game_start and win_text:
      screen.blit(win_text, (200, 330))
    for i in range(22,800,54):
        if i == 22 or i == 778:
            pygame.draw.line(screen,line_color,[i,22],[i,778],4)
        else:
            pygame.draw.line(screen,line_color,[i,22],[i,778],2)
        if i == 22 or i == 778:
            pygame.draw.line(screen,line_color,[22,i],[778,i],4)
        else:
            pygame.draw.line(screen,line_color,[22,i],[778,i],2)
    pygame.draw.circle(screen, line_color,[400,400], 8,0)
    pygame.draw.circle(screen, line_color,[184,184], 8,0)
    pygame.draw.circle(screen, line_color,[616,616], 8,0)
    pygame.draw.circle(screen, line_color,[184,616], 8,0)
    pygame.draw.circle(screen, line_color,[616,184], 8,0)

#判断鼠标的位置是否位于棋盘内部
def is_valid(x,y):
    if x > 800 or y > 800:
      return False
    return True

#根据鼠标的位置定位到棋盘上的点
def find_pos(x,y):
    for i in range(22,800,54):
        for j in range(22,800,54):
            if x >= i - 27 and x <= i + 27 and y >= j - 27 and y <= j + 27:
                return i,j
    return x,y

def check_exist(x,y):
    global exist_chess
    for val in exist_chess:
        if val[0][0] == x and val[0][1] == y:
            return True
    return False

def get_empty():
    empty_position = []
    for i in range(15):
        for j in range(15):
            if board[i][j] == 0:
                score = position[i][j]
                empty_position.append((score,i,j))
    empty_position.sort(reverse=True)
    return empty_position

def add_chess():
    global turn
    global flag
    global tim
    x,y = pygame.mouse.get_pos()
    x,y = find_pos(x,y)
    if is_valid(x,y) and not check_exist(x,y) and game_start:
      pygame.draw.rect(screen,rect_color,[x-20,y-20,40,40],2,1)
    mouse_pressed = pygame.mouse.get_pressed()
    if mouse_pressed[0] and tim==0:
        flag=True
        if not check_exist(x,y) and is_valid(x,y) and game_start and turn == player_num:#判断是否可以落子，再落子
            if len(exist_chess)%2==0:#黑子
                exist_chess.append([[x,y],black_chess_color])
            else:
                exist_chess.append([[x,y],white_chess_color])
            if turn == 1:
              turn = 2
            else:
              turn = 1
    
    #鼠标左键延时作用,延时200ms
    if flag:
        tim += 1
    if tim == 200:
        flag = False
        tim = 0

    for val in exist_chess:
        pygame.draw.circle(screen, val[1],val[0], 20,0)

def update_chess():
  global exist_chess    
  for pos, color in exist_chess:
    x = (pos[0] - 22) // 54  
    y = (pos[1] - 22) // 54
    if color == black_chess_color:  
      board[x][y] = 1  
    elif color == white_chess_color:  
      board[x][y] = 2 

def check_win():
  for dx, dy in directions:  
    for x in range(15):  
      for y in range(15 - abs(dy)):  # 根据方向调整y的范围  
        temp_count = 1  
        for step in range(1, 5):    
          nx, ny = x + step * dx, y + step * dy 
          if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] and board[x][y] and board[nx][ny] == board[x][y]:  
            temp_count += 1  
          else:
            break  
        if temp_count == 5:  
          return board[x][y]    
  return 0  # 无人获胜

#空 = 0,
#眠二 = 1,
#活二 = 2,
#眠三 = 3
#活三 = 4,
#冲四 = 5,
#活四 = 6,
#活五 = 7,

def getScore(player_chess, enemy_chess):
        player_score = 0
        enemy_score = 0
        if player_chess[7] > 0:
            return (50000, 0)
        
        if enemy_chess[7] > 0:
            return (0, 50000)

        if enemy_chess[6] > 0:
            return (0, 10000)
        
        if enemy_chess[5] > 0:
            return (0, 9500)
        
        if player_chess[6] > 0 or player_chess[5] >= 2:
            return (9300, 0)
        
        if player_chess[5] > 0 and player_chess[4] > 0:
            return (9200, 0)
        
        if enemy_chess[4] > 0 and player_chess[5] == 0:
            return (0, 9100)
        
        if (player_chess[4] > 1 and enemy_chess[4] == 0 and enemy_chess[3] == 0):
            return (9000, 0)
        
        if player_chess[5] > 0:
            player_score += 2000

        if player_chess[4] > 1:
            player_score += 500
        elif player_chess[4] > 0:
            player_score += 100

        if enemy_chess[4] > 1:
            enemy_score += 2000
        elif enemy_chess[4] > 0:
            enemy_score += 400

        if player_chess[3] > 0:
            player_score += player_chess[3] * 10

        if enemy_chess[3] > 0:
            enemy_score += enemy_chess[3] * 10

        if player_chess[2] > 0:
            player_score += player_chess[2] * 4

        if enemy_chess[2] > 0:
            enemy_score += enemy_chess[2] * 4

        if player_chess[1] > 0:
            player_score += player_chess[1] * 4

        if enemy_chess[1] > 0:
            enemy_score += enemy_chess[1] * 4

        return (player_score, enemy_score)

def getLine(x, y, direction, me, you):
  global board
  line = [0 for _ in range(9)]
  next_x = x + (-5 * direction[0])
  next_y = y + (-5 * direction[1])
  for i in range(9):
    next_x += direction[0]
    next_y += direction[1]
    if (next_x >= 15 or next_x < 0 or next_y < 0 or next_y >= 15):
      line[i] = you  
    else:
      line[i] = board[next_x][next_y]
  return line


def record_move(x, y, left, right, dir_index, direction):
    global record
    x_ = x + (-5 + left) * direction[0]
    y_ = y + (-5 + left) * direction[1]
    for i in range(left, right):
        x_ += direction[0]
        y_ += direction[1]
        record[x_][y_][dir_index] = 1

def evaluate_type(x,y,dir_num,dir,me,you):
    global count
    line = getLine(x, y, dir, me, you)
    l_index = 4
    r_index = 4
    while r_index < 8:
        if line[r_index + 1] != me:
          break
        r_index += 1
    while l_index > 0:
        if line[l_index - 1] != me:
          break
        l_index -= 1
    l_range = l_index
    r_range = r_index
    while r_range < 8:
        if line[r_range + 1] == you:
            break
        r_range += 1
    while l_range > 0:
        if line[l_range - 1] == you:
            break
        l_range -= 1
    chess_range = r_range - l_range + 1
#chess_range < 5说明该方向上左右两边两个敌方的子中间夹着最多四个子，这种情况不属于任何棋型
    if chess_range < 5:
        record_move(x, y, l_range, r_range, dir_num, dir)
        return 0
    record_move(x, y, l_index, r_index, dir_num, dir)
    m_range = r_index - l_index + 1
    if m_range == 5:
        count[me-1][7] += 1

    if m_range == 4:
        #两边都为空则为活四 XMMMMX
        if line[l_index - 1] == 0 and line[r_index + 1] == 0:
          count[me-1][6] += 1
        #一边为空另一边为敌方棋子则为冲四 XMMMMP PMMMMX
        elif line[l_index - 1] == 0 or line[r_index + 1] == 0:
          count[me-1][5] += 1
    
    if m_range == 3:
        if line[l_index - 1] == 0:
            if line[l_index - 2] == me:  #冲四 MXMMM
                record_move(x, y, l_index - 2, l_index - 1, dir_num, dir)
                count[me-1][5] += 1
        if line[r_index + 1] == 0:
            if line[r_index + 2] == me:  #冲四 MMMXM
                record_move(x, y, r_index + 1, r_index + 2, dir_num, dir)
                count[me-1][5] += 1
        if line[l_index - 2] == me or line[r_index + 2] == me:
            pass
        elif line[l_index - 1] == 0 and line[r_index + 1] == 0:
            if chess_range > 5:  #活三 XMMMXX XXMMMX
                count[me-1][4] += 1
            else:  #眠三 PXMMMXP
                count[me-1][3] += 1
        elif line[l_index - 1] == 0 or line[r_index + 1] == 0:  #眠三 PMMMX XMMMP
            count[me-1][3] += 1
        
    if m_range == 2:
        left_three = False
        right_three = False
        if line[l_index - 1] == 0:
            if line[l_index - 2] == me:
                record_move(x, y, l_index - 2, l_index - 1, dir_num, dir)
                if line[l_index - 3] == 0:
                    if line[r_index + 1] == 0:  #活三 XMXMMX
                        count[me-1][4] += 1
                    else:  #眠三 XMXMMP
                        count[me-1][3] += 1
                    left_three = True
                elif line[l_index - 3] == you:  
                    if line[r_index + 1] == 0:  #眠三 PMXMMX
                        count[me-1][3] += 1
                    left_three = True
        if line[r_index + 1] == 0:
            if line[r_index + 2] == me:
                if line[r_index + 3] == me:  #冲四 MMXMM
                    record_move(x, y, r_index + 1, r_index + 2, dir_num, dir)
                    count[me-1][5] += 1
                    right_three = True
                elif line[r_index + 3] == 0:
                    if line[l_index - 1] == 0:  #活三 XMMXMX
                        count[me-1][4] += 1
                    else:  #眠三 PMMXMX
                        count[me-1][3] += 1
                    right_three = True
                elif line[l_index - 1] == 0:  #眠三 XMMXMP
                    count[me-1][3] += 1
                    right_three = True
        if left_three or right_three:
            pass
        elif line[l_index - 1] == 0 and line[r_index + 1] == 0:  #活二 XMMX
            count[me-1][2] += 1
        elif line[l_index - 1] == 0 or line[r_index + 1] == 0:  #眠二 PMMX, XMMP
            count[me-1][1] += 1

    if m_range == 1:
        if line[l_index - 1] == 0:
          if line[l_index - 2] == me and line[l_index - 3] == 0 and line[r_index + 1] == you: #眠二 XMXMP
              count[me-1][1] += 1 
          if line[r_index + 1] == 0:
            if line[r_index + 2] == me:
              if line[r_index + 3] == 0:
                if line[l_index-1]==0:  #活二 XMXMX
                  count[me-1][2] += 1
                else:  #眠二 PMXMX
                  count[me-1][1] += 1
            elif line[r_index + 2] == 0:
              if line[r_index + 3] == me and line[r_index + 4] == 0:  #眠二 XMXXMX
                count[me-1][2] += 1

    # 以上都不是则为none棋型
    return 0


def evaluatePoint(x, y, me, you):
  global record
  global direction
  for i in range(4):
    if record[x][y][i] == 0:
      evaluate_type(x, y, i, direction[i], me, you) #计算棋型的函数

def evaluate(current_turn):
  global record
  global count
  record = [[[0,0,0,0]for _ in range(15)] for _ in range(15)]
  count = [[0 for _ in range(8)] for _ in range(2)]
  if current_turn == 1:
    player = 1
    enemy = 2
  else:
    player = 2
    enemy = 1
  for x in range(15):
    for y in range(15):
      if board[x][y] == player:
        evaluatePoint(x, y, player, enemy)
      elif board[x][y] == enemy:
        evaluatePoint(x, y, enemy, player)
  playerchess = count[player-1]
  enemychess = count[enemy-1]
  mscore, oscore = getScore(playerchess, enemychess)
  return (mscore - oscore)

def alpha_beta_search(depth, alpha, beta, is_maximizing,current_turn):  
    global board  
    if depth == 0:  
        return evaluate(3-current_turn), None, None  
    moves = get_empty()  
    best_move = None  
    best_score = -99999 if is_maximizing else 99999    
    for score_heuristic, x, y in moves:  
        board[x][y] = current_turn       
        score, move_x, move_y = alpha_beta_search(depth - 1, alpha, beta, not is_maximizing,3-current_turn)    
        board[x][y] = 0  # 回溯    
        if is_maximizing:  
            if score > best_score:  
                best_score = score  
                best_move = (x, y)  
                alpha = max(alpha, score)  
                if beta <= alpha:  # 剪枝 
                    break  
        else:  
            if score < best_score:  
                best_score = score  
                best_move = (x, y)  
                beta = min(beta, score)  
                if beta <= alpha:  # 剪枝  
                    break    
    return best_score, best_move[0], best_move[1]  
  
def search():  
    global turn  
    global board  
    depth = 1  
    alpha = -99999  
    beta = 99999
    score, x, y = alpha_beta_search(depth, alpha, beta, True,turn)  
    return (score, x, y)

def AI():
  max_score,ai_x,ai_y = search()
  if ai_num == 1:
    exist_chess.append([[ai_x*54+22,ai_y*54+22],black_chess_color])
  else:
    exist_chess.append([[ai_x*54+22,ai_y*54+22],white_chess_color])
  return

while True:
    for event in pygame.event.get():
        if event.type in (QUIT,KEYDOWN):
            sys.exit()
        elif event.type == MOUSEBUTTONDOWN:
            if button_black.collidepoint(event.pos) and not game_start:
              player_num = 1
              ai_num = 2
              exist_chess.clear()
              game_start = True
              board = np.zeros((15, 15), dtype=int)
              win_text = None
              print("Choose Black!")
            elif button_white.collidepoint(event.pos) and not game_start:
              player_num = 2
              ai_num = 1
              exist_chess.clear()
              game_start = True
              board = np.zeros((15, 15), dtype=int)
              win_text = None
              print("Choose White!")
            elif button_restart.collidepoint(event.pos):
              turn = 1
              game_start = False
              exist_chess.clear()
              board = np.zeros((15, 15), dtype=int)
              win_text = None
              print("Restart!")

    draw_map()

    if game_start:
      if turn == ai_num:
        AI()
        if turn == 1:
          turn = 2
        else:
          turn = 1

    add_chess()

    update_chess()

    winner = check_win()

    if winner: 
        if winner == 1:  
            win_text = font.render("Black Wins!", True, (255, 0, 0))
            game_start = False
            
        else:  
            win_text = font.render("White Wins!", True, (255, 0, 0))
            game_start = False
    pygame.display.update()

