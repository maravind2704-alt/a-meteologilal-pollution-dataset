#include <stdio.h>
#include <windows.h>
#include <stdlib.h>

int score = 0;

void draw()
{
    system("cls");

    printf("=============================\n");
    printf("      ENDLESS RUNNER\n");
    printf("=============================\n\n");

    printf("Score : %d\n\n",score);

    printf("       O\n");
    printf("      /|\\\n");
    printf("      / \\\n");
    int playerX = 20;
if(GetAsyncKeyState(VK_LEFT))
    playerX--;

if(GetAsyncKeyState(VK_RIGHT))
    playerX++;
    for(int i=0;i<playerX;i++)
    printf(" ");

printf("O");
int obstacleX=15;
int obstacleY=0;
obstacleY++;
obstacleY=0;

obstacleX=rand()%40;

}

int main()
{
    while(1)
    {
        draw();

        score++;

        Sleep(100);
    }
}