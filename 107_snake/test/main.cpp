#include <iostream>
#include <algorithm>
#include <unistd.h>
#include <thread>
#include <termios.h>

using namespace std;

struct termios t;

int SIZE_X = 80;
int SIZE_Y = 30;

void redraw(int *x, int *y)
{
    for (int i = 0; i < SIZE_Y; i++)
    { // clear screen
        cout << endl;
    }
    for (int i = 0; i < (*x); i++)
    { // clear screen
        cout << " ";
    }
    cout << "x" << endl;
    for (int i = 0; i < (*y); i++)
    { // clear screen
        cout << endl;
    }
}

void thread_job(bool *running, int *x, int *y)
{
    while (*running)
    {
        (*x)++;
        (*x) = min((*x), SIZE_X);
        usleep(500000);
        redraw(x, y);
    }
}

int main()
{
    tcgetattr(STDIN_FILENO, &t);
    t.c_lflag &= ~ICANON;
    tcsetattr(STDIN_FILENO, TCSANOW, &t);

    int x = 0;
    int y = 0;
    bool running = true;

    thread t1(thread_job, &running, &x, &y);

    // t1.join();
    // char user_input;

    while (running)
    {
        char user_input = getchar();
        redraw(&x, &y);
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