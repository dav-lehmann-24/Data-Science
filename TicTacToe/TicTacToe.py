# TIC-TAC-TOE GAME (Group members: David, Niki)

# Function to print the board
def printBoard(board):
    print("\n")
    print(board[0] + " | " + board[1] + " | " + board[2])
    print("--+---+--")
    print(board[3] + " | " + board[4] + " | " + board[5])
    print("--+---+--")
    print(board[6] + " | " + board[7] + " | " + board[8])
    print("\n")

# Function to check for a win
def checkWinner(board, symbol):
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for (a, b, c) in wins:
        # print(f"Checking positions {a}, {b}, {c} for symbol {symbol}")
        if board[a] == board[b] == board[c] == symbol:
            # print("YES")
            return True
    return False

# Main game
def main():
    board = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    Player1 = "X"
    Player2 = "O"
    currentPlayer = Player1
    active_game = True

    turns = 0
    while active_game:
        while turns < 9:
            printBoard(board)

            try:
                move = int(input("Player " + currentPlayer + ", enter your move (1-9): "))
                if move < 1 or move > 9:
                    print("Invalid move. Try again.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 9.")
                continue

            if board[move-1] == "X" or board[move-1] == "O":
                print("Invalid move. Try again.")
                continue
            else:
                board[move-1] = currentPlayer
                turns += 1

                if checkWinner(board, currentPlayer):
                    printBoard(board)
                    print("Player " + currentPlayer + " wins!")
                    response = input("Do you want to play again? (y/n): ").strip().lower()
                    if response == 'y':
                        board = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
                        currentPlayer = Player1
                        turns = 0
                        continue
                    elif response == 'n':
                        print("Thanks for playing!")
                        active_game = False
                        break
                    else:
                        print("Invalid input. Exiting game.")
                        active_game = False
                        break

                currentPlayer = Player2 if currentPlayer == Player1 else Player1

        if not active_game:
            break

        printBoard(board)
        print("It's a draw!")
        response = input("Do you want to play again? (y/n): ").strip().lower()
        if response == 'y':
            board = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            currentPlayer = Player1
            turns = 0
            continue
        else:
            print("Thanks for playing!")
            active_game = False

main()