/*
Basic implementation of Conway's game of life. Read in RLE file
from rats.rle

Kewei Han
*/
#include <SimpleECS.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <filesystem>


using namespace std;
using namespace SimpleECS;

const string RLE_PATH = "assets/rats.rle";
const int SCREEN_HEIGHT = 960;
const int SCREEN_WIDTH = 1480;
const int CELL_SIZE = 1; // SIZE in pixels of visible cells

const double GEN_LENGTH = 0.05; // Time in seconds per generation

std::vector<bool> parseRLELine(const std::string& line) {
    std::vector<bool> row;
    int count = 0;

    for (char c : line) {
        if (isdigit(c)) {
            count = count * 10 + (c - '0');
        }
        else {
            if (c == '!') { break; }
            if (c == '\n') { continue; }
            bool state = (c == 'o');
            for (int i = 0; i < max(1, count); ++i) {
                row.push_back(state);
            }
            count = 0;
        }
    }
    return row;
}

std::vector<std::vector<bool>> parseRLE(const std::string& filePath) {
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "conway: Unable to find or open rle file. Make sure to execute from executable directory.\n";
        exit(1); // exit with error code
    }

    // Skip headers
    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == 'x') {
            break;
        }
    }

    int x, y;
    sscanf(line.c_str(), "x = %d, y = %d", &x, &y); // parse the size of the grid

    std::vector<std::vector<bool>> grid; // create a 2D array

    int row = 0, col = 0;
    std::string cellData;
    while (std::getline(file, cellData, '$')) {

        int numEmpties = 0;
        if (isdigit(cellData.back())) {
            numEmpties = cellData.back() - '0' - 1;
            cellData.pop_back();
        }

        // Parse the remaining line
        if (!cellData.empty()) {
            grid.push_back(parseRLELine(cellData));
        }

        for (int i = 0; i < numEmpties; ++i) {
            grid.push_back(std::vector<bool>());
        }
  
    }

    file.close();
    std::reverse(grid.begin(), grid.end());
    return grid;
}

// Component for controlling paddle movement manually
class Cell : public Component {
public:
    static int viewGridWidth;
    static int viewGridHeight;
    static int cellSize;
    static int liveCells;
    static vector<vector<bool>> cells;

    Cell(int _r, int _c) : r(_r), c(_c) {}
    
    void initialize() 
    {
        double xPos = c * cellSize - SCREEN_WIDTH / 2;
        double yPos = r * cellSize - SCREEN_HEIGHT / 2;
        entity->transform->position = { xPos, yPos };
        rr = entity->getComponent<RectangleRenderer>();
    }

    void update() override
    {
        rr->setActive(cells[r][c]);
    }

    void calcNextGen()
    {
        // conway rules - 23/3
        
        // get number of neighbors
        int liveNeighbors = 0;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr != 0 || dc != 0) {  // Exclude the cell itself
                    // Toridial neighbor calculation to handle edges
                    int nr = (r + dr + cells.size()) % cells.size();
                    int nc = (c + dc + cells[0].size()) % cells[0].size();
                    liveNeighbors += cells[nr][nc] ? 1 : 0;
                }
            }
        }

        // Any live cell with fewer than two live neighbors dies, as if by underpopulation.
        if (cells[r][c] && liveNeighbors < 2)
        {
            nextGenActive = false;
        }
        // Any live cell with two or three live neighbors lives on to the next generation.
        else if (cells[r][c] && (liveNeighbors == 3 || liveNeighbors == 2))
        {
            nextGenActive = true;
        }
        // Any live cell with more than three live neighbors dies, as if by overpopulation.
        else if (cells[r][c] && liveNeighbors > 3)
        {
            nextGenActive = false;
        }
        // Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
        else if (!cells[r][c] && liveNeighbors == 3)
        {
            nextGenActive = true;
        }
    }

    void transitionNextGen()
    {
        cells[r][c] = nextGenActive;
    }

private:
    int r, c;
    bool nextGenActive = false;
    Handle<RectangleRenderer> rr;
};

int Cell::liveCells                 = 0;
int Cell::viewGridWidth             = SCREEN_WIDTH / CELL_SIZE;
int Cell::viewGridHeight            = SCREEN_HEIGHT / CELL_SIZE;
int Cell::cellSize                  = CELL_SIZE;
vector<vector<bool>> Cell::cells    = vector<vector<bool>>(viewGridHeight + 1, vector<bool>(viewGridWidth + 1, false));

class CellManager : public Component {
public:
    CellManager(vector<vector<bool>>& _parsedGrid): parsedGrid(_parsedGrid) {}

    void initialize()
    {
        // Create entities from parsedGrid
        auto scene = Game::getInstance().getCurrentScene();
        for (int r = 0; r < Cell::viewGridHeight + 1; r++)
        {
            for (int c = 0; c < Cell::viewGridWidth + 1; c++)
            {
                auto cellEnt = scene->createEntity();
                cellEnt->addComponent<RectangleRenderer>(CELL_SIZE, CELL_SIZE, Color(0xFF, 0xFF, 0xFF));
                cellEnt->addComponent<Cell>(r, c);
            }
        }

        // Copy parsedGrid into center of scene grid
        int top = (Cell::cells.size() - parsedGrid.size()) / 2;
        int left = (Cell::cells[0].size() - parsedGrid[0].size()) / 2;
        for (int r = top, i = 0; r < Cell::cells.size() && i < parsedGrid.size(); ++r, ++i)
        {
            for (int c = left, j = 0; c < Cell::cells[0].size() && j < parsedGrid[i].size(); ++c, ++j)
            {
                Cell::cells[r][c] = parsedGrid[i][j];
            }
        }
    }

    void update() override
    {
        timer += Timer::getDeltaTime();
        if (timer >= GEN_LENGTH * 1000)
        {
            timer = 0;
            auto scene = Game::getInstance().getCurrentScene();
            for (auto& cell : *scene->getComponents<Cell>())
            {
                cell.calcNextGen();
            }

            for (auto& cell : *scene->getComponents<Cell>())
            {
                cell.transitionNextGen();
            }

            generation++;
        }
    }

    static int generation;

private: 
    double timer = 0;
    vector<vector<bool>> parsedGrid;
};

int CellManager::generation = 0;

class GenerationCounter : public Component {
public:

    void initialize() {
        textRender = entity->getComponent<FontRenderer>();
        entity->transform->position = Vector(0, 0);
    };

    void update() {
        string text = "Generation: " + std::to_string(CellManager::generation);
        textRender->text = text;
    }

    uint64_t framesPassed = 0;
    Handle<FontRenderer> textRender;
};

class AvgFrameCounter : public Component {
public:

    void initialize() {
        textRender = entity->getComponent<FontRenderer>();
        entity->transform->position = Vector(0, 25);
    };

    void update() {
        framesPassed++;
        int64_t lifeTime = Timer::getProgramLifetime();
        int64_t avgFPS = framesPassed / std::max(static_cast<int>(Timer::getProgramLifetime()) / 1000, 1);
        string text = "Average FPS: " + std::to_string(avgFPS);

        textRender->text = text;
    }

    uint64_t framesPassed = 0;
    Handle<FontRenderer> textRender;
};


int main() {
    auto parsedGrid = parseRLE(RLE_PATH);
     //print the grid
     //for (const auto& row : parsedGrid) {
     //    for (bool cell : row) {
     //        std::cout << (cell ? 'o' : 'b');
     //    }
     //    std::cout << '\n';
     //}
    // Create Scene
    Scene* scene = new Scene(Color(0, 0, 0, 255));
    Game::getInstance().addScene(scene);
    scene->createEntity()->addComponent<CellManager>(parsedGrid);
    
    auto genDisplay = scene->createEntity();
    genDisplay->addComponent<FontRenderer>("Default", "assets/bit9x9.ttf", 26, Color(124, 200, 211, 0xff));
    genDisplay->addComponent<GenerationCounter>();

    // TODO: fix this bug. If BoxCollider isn't present library crashses.
    auto dummy = scene->createEntity();
    dummy->addComponent<BoxCollider>();

    // TODO: fix this bug. If addComponent for a type is first called in initialize, there is
    // potential for a crash (due to vector invalidation)
    dummy->addComponent<RectangleRenderer>(0,0);
    dummy->addComponent<Cell>(0, 0);
    
    RenderConfig config;
    config.width = SCREEN_WIDTH;
    config.height = SCREEN_HEIGHT;
    config.gameName = "Conway";

    Game::getInstance().configure(config);
    Game::getInstance().startGame();

    return 0;
}