#include <SDL.h>
#include <cmath>
#include <algorithm>
#include <iostream>

// g++ -std=c++11 -Wall -O0 -g src/main.cpp $(pkg-config --cflags --libs sdl2) -o build/debug/plot
// ./build/debug/plot

inline int toScreenX(double x, double xmin, double xmax, int W) {
    return (int) std::lround((x - xmin) * (W - 1) / (xmax - xmin));
}

inline int toScreenY(double y, double ymin, double ymax, int H) {
    return (int) std::lround((ymax - y) * (H - 1) / (ymax - ymin));
}

inline int clamp(int v, int low, int hi) {
    return v < low ? low : (v > hi ? hi : v);
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

    double xmin = -10;
    double xmax = 10;

    double ymin = -2;
    double ymax = 2;

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
        }

        // Clear
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Draw axes
        SDL_SetRenderDrawColor(renderer, 80, 80, 80, 255);
        if (xmin <= 0 && 0 <= xmax) {
            int x0 = toScreenX(0, xmin, xmax, width);
            SDL_RenderDrawLine(renderer, x0, 0, x0, height - 1);
        }

        if (ymin <= 0 && 0 <= ymax) {
            int y0 = toScreenY(0, ymin, ymax, height);
            SDL_RenderDrawLine(renderer, 0, y0, width - 1, y0);
        }

        // Plot function
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

        bool havePrev = false;
        int pxPrev = 0, pyPrev = 0;

        for (int sx = 0; sx < width; sx++) {
            double x = xmin + (xmax - xmin) * (double)sx / (double)(width - 1);
            double y = function(x); // y = f(x)

            // Skip non-finite results (asymptotes, domain issues)
            if (!std::isfinite(y)) { havePrev = false; continue; }

            int sy = toScreenY(y, ymin, ymax, height);

            // If y is outside viewport, break the line so it doesn’t smear
            if (sy < -100000 || sy > 100000) { havePrev = false; continue; }

            if (havePrev) {
                // Clip a bit to avoid crazy long lines at edges
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