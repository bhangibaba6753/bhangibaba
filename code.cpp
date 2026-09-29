#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>

using namespace std;

struct City {
    int id;
    double x, y;
};

vector<City> cities = {
    {1, 60, 200}, {2, 23, 45}, {3, 15, 150}, {4, 85, 90}, {5, 71, 123},
    {6, 98, 45}, {7, 50, 220}, {8, 20, 30}, {9, 40, 180}, {10, 75, 155},
    {11, 82, 60}, {12, 91, 120}, {13, 12, 210}, {14, 37, 100}, {15, 28, 77},
    {16, 66, 89}, {17, 55, 130}, {18, 73, 140}, {19, 88, 33}, {20, 42, 170}
};

double distMatrix[20][20];

struct Individual {
    vector<int> tour;
    double distance;
    double fitness;
};

void computeDistanceMatrix() {
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {
            double dx = cities[i].x - cities[j].x;
            double dy = cities[i].y - cities[j].y;
            distMatrix[i][j] = sqrt(dx * dx + dy * dy);
        }
    }
}

double calculateTotalDistance(const vector<int>& tour) {
    double total = 0;
    for (size_t i = 0; i < tour.size() - 1; ++i) {
        total += distMatrix[tour[i]][tour[i+1]];
    }
    total += distMatrix[tour.back()][tour.front()];
    return total;
}

void evaluatePopulation(vector<Individual>& pop) {
    for (auto& ind : pop) {
        ind.distance = calculateTotalDistance(ind.tour);
        ind.fitness = 1.0 / ind.distance;
    }
    sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
        return a.fitness > b.fitness;
    });
}

Individual rouletteWheelSelection(const vector<Individual>& pop, mt19937& gen) {
    double totalFitness = 0;
    for (const auto& ind : pop) totalFitness += ind.fitness;
    uniform_real_distribution<> dis(0, totalFitness);
    double r = dis(gen);
    double cumulative = 0;
    for (const auto& ind : pop) {
        cumulative += ind.fitness;
        if (cumulative >= r) return ind;
    }
    return pop.back();
}

Individual orderCrossover(const Individual& p1, const Individual& p2, mt19937& gen) {
    int n = p1.tour.size();
    uniform_int_distribution<> dis(0, n - 1);
    int start = dis(gen);
    int end = dis(gen);
    if (start > end) swap(start, end);

    Individual offspring;
    offspring.tour.assign(n, -1);
    vector<bool> inOffspring(n, false);

    for (int i = start; i <= end; ++i) {
        offspring.tour[i] = p1.tour[i];
        inOffspring[p1.tour[i]] = true;
    }

    int p2Index = 0;
    for (int i = 0; i < n; ++i) {
        if (offspring.tour[i] == -1) {
            while (inOffspring[p2.tour[p2Index]]) {
                p2Index++;
            }
            offspring.tour[i] = p2.tour[p2Index];
            inOffspring[p2.tour[p2Index]] = true;
        }
    }
    return offspring;
}

void swapMutation(Individual& ind, mt19937& gen) {
    uniform_real_distribution<> prob(0.0, 1.0);
    if (prob(gen) < 0.1) {
        uniform_int_distribution<> dis(0, ind.tour.size() - 1);
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        swap(ind.tour[idx1], ind.tour[idx2]);
    }
}

Individual runGA(int popSize, int numGenerations) {
    random_device rd;
    mt19937 gen(rd());

    vector<Individual> population(popSize);
    vector<int> baseTour(20);
    iota(baseTour.begin(), baseTour.end(), 0);

    for (int i = 0; i < popSize; ++i) {
        population[i].tour = baseTour;
        shuffle(population[i].tour.begin(), population[i].tour.end(), gen);
    }

    evaluatePopulation(population);

    for (int genIdx = 0; genIdx < numGenerations; ++genIdx) {
        vector<Individual> newPopulation;
        newPopulation.push_back(population[0]);

        while (newPopulation.size() < popSize) {
            Individual p1 = rouletteWheelSelection(population, gen);
            Individual p2 = rouletteWheelSelection(population, gen);
            Individual offspring = orderCrossover(p1, p2, gen);
            swapMutation(offspring, gen);
            newPopulation.push_back(offspring);
        }
        population = newPopulation;
        evaluatePopulation(population);
    }
    return population[0];
}

int main() {
    computeDistanceMatrix();

    vector<pair<int, int>> configs = {
        {10, 100}, {20, 100}, {30, 100},
        {10, 200}, {20, 200}, {30, 200}
    };

    for (auto config : configs) {
        int popSize = config.first;
        int gens = config.second;
        Individual best = runGA(popSize, gens);

        cout << "Population: " << popSize << ", Generations: " << gens 
             << " | Distance: " << fixed << setprecision(2) << best.distance 
             << " | Tour: ";
        for (int cityIdx : best.tour) {
            cout << (cityIdx + 1) << " ";
        }
        cout << "\n";
    }

    return 0;
}