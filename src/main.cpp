#include <SDL.h>
#include <cmath>
#include <algorithm>
#include <iostream>

// g++ -std=c++11 -Wall -O0 -g src/main.cpp $(pkg-config --cflags --libs sdl2) -o build/debug/plot
// ./build/debug/plot

int toScreenX(double x, double xMin, double xMax, int width) {
    return (int) std::lround((x - xMin) * (width - 1) / (xMax - xMin));
}

int toScreenY(double y, double yMin, double yMax, int height) {
    return (int) std::lround((yMax - y) * (height - 1) / (yMax - yMin));
}

int clamp(int val, int low, int high) {
    return val < low ? low : (val > high ? high : val);
}

double function (double x) {
    return std::sin(x);
};

int main() {
    const int width = 800;
    const int height = 600;

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;

    SDL_CreateWindowAndRenderer(width, height, 0, &window, &renderer);
    SDL_SetWindowTitle(window, "Plot");

    double xMin = -10;
    double xMax = 10;

    double yMin = -2;
    double yMax = 2;

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Draw axes
        SDL_SetRenderDrawColor(renderer, 80, 80, 80, 255);
        
        if (xMin <= 0 && 0 <= xMax) {
            int x0 = toScreenX(0, xMin, xMax, width);
            SDL_RenderDrawLine(renderer, x0, 0, x0, height - 1);
        }

        if (yMin <= 0 && 0 <= yMax) {
            int y0 = toScreenY(0, yMin, yMax, height);
            SDL_RenderDrawLine(renderer, 0, y0, width - 1, y0);
        }

        // Plot function
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

        bool havePrev = false;
        int pxPrev = 0, pyPrev = 0;

        for (int sx = 0; sx < width; sx++) {
            double x = xMin + (xMax - xMin) * (double) sx / (double)(width - 1);
            double y = function(x); // y = f(x)

            // Skip non-finite results, such as asymptotes
            if (!std::isfinite(y)) { havePrev = false; continue; }

            int sy = toScreenY(y, yMin, yMax, height);

            // If y is outside viewport, break the line 
            if (sy < -100000 || sy > 100000) { havePrev = false; continue; }

            if (havePrev) {
                // Clamping to avoid long lines at edges
                int y1 = clamp(pyPrev, -2000, height + 2000);
                int y2 = clamp(sy, -2000, height + 2000);
                SDL_RenderDrawLine(renderer, pxPrev, y1, sx, y2);
            }

            pxPrev = sx;
            pyPrev = sy;
            havePrev = true;
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16); // 60fps
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}