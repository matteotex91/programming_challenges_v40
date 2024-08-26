#include <iostream>
#include <algorithm>
#include <unistd.h>
#include <thread>
#include <termios.h>

using namespace std;

struct termios t;

const int SIZE_X = 80;
const int SIZE_Y = 30;

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

void thread_job(bool *running, int (*map)[SIZE_X][SIZE_Y])
{
    int x = 0;
    while (*running)
    {
        x = (x + 1) % SIZE_X;
        *map[x][0] = 1;
        usleep(500000);
        redraw(map);
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

    int x = 0;
    int y = 0;
    bool running = true;

    thread t1(thread_job, &running, &x, &y);

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