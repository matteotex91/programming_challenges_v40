#include <iostream>
#include <algorithm>
#include <unistd.h>
#include <thread>
#include <termios.h>
#include <random>

using namespace std;

struct termios t;

const int SIZE_X = 80;
const int SIZE_Y = 30;
const int INIT_SIZE = 3;
const int FOOD_INDEX = SIZE_X * SIZE_Y;
const int INIT_DIRECTION = 0;

void redraw(int (*map)[SIZE_X][SIZE_Y])
{
    cout << "+";
    for (int ix = 0; ix < SIZE_X; ix++)
    {
        cout << "-";
    }
    cout << "+" << endl;
    for (int iy = 0; iy < SIZE_Y; iy++)
    {
        cout << "|";
        for (int ix = 0; ix < SIZE_X; ix++)
        {
            switch (*map[ix][iy])
            {
            case 0:
                cout << " ";
                break;

            default:
                cout << "0";
                break;
            }
        }
        cout << "|" << endl;
    }
    cout << "+";
    for (int ix = 0; ix < SIZE_X; ix++)
    {
        cout << "-";
    }
    cout << "+" << endl;
}

void main_cycle(bool *running, int (*map)[SIZE_X][SIZE_Y], int *head_x, int *head_y, int *snake_size, int *food_x, int *food_y)
{
    random_device rd;  // obtain a random number from hardware
    mt19937 gen(rd()); // seed the generator
    uniform_int_distribution<> food_rand_X(0, SIZE_X - 1);
    uniform_int_distribution<> food_rand_Y(0, SIZE_Y - 1);
    bool alive = true;
    while (*running)
    {
        for (int ix = 0; ix < SIZE_X; ix++)
        {
            for (int iy = 0; iy < SIZE_Y; iy++)
            {
                *map[ix][iy] = 0;
            }
        }
        *head_x = SIZE_X / 2;
        *head_y = SIZE_Y / 2;
        *snake_size = INIT_SIZE;
        for (int index = 0; index < *snake_size; index++)
        {
            *map[*head_x - index][*head_y] = *snake_size - index; // ??? check
        }
        *food_x = food_rand_X(gen);
        *food_y = food_rand_Y(gen);
        *map[*food_x][*food_y] = FOOD_INDEX;
        while (alive)
        {
            // check if stepped on the food
            if ((*food_x == *head_x) && (*food_y == *head_y))
            {
                *snake_size++;
            }
            // make step
            switch ()
        }
    }
}

int main()
{
    tcgetattr(STDIN_FILENO, &t);
    t.c_lflag &= ~ICANON;
    tcsetattr(STDIN_FILENO, TCSANOW, &t);

    int map[SIZE_X][SIZE_Y] = {{}};

    for (int ix = 0; ix < SIZE_X; ix++)
    {
        for (int iy = 0; iy < SIZE_Y; iy++)
        {
            map[ix][iy] = 0;
        }
    }

    int head_x = 0;
    int head_y = 0;
    bool running = true;

    thread t1(main_cycle, &running, &head_x, &head_y);

    // t1.join();
    // char user_input;

    while (running)
    {
        char user_input = getchar();
        redraw(&map);
        // cin >> user_input;
        switch (user_input)
        {
        case 'w':
            y++;
            break;
        case 's':
            y--;
            break;
        case 'd':
            x++;
            break;
        case 'a':
            x--;
            break;
        case 27: // ESC
            running = false;
            break;

        default:
            break;
        }
        x = min(SIZE_X, max(x, 0));
        y = min(SIZE_Y, max(y, 0));
    }
    return 0;
}