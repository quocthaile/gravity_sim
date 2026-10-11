#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>

using namespace std;

struct Position
{
    double x;
    double y;
    double z;
};

using Key = tuple<double, int, int>;
// Key = {timeIndex, step, objectIndex}

bool ReadCSV(const string &filename, map<Key, Position> &positions)
{
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Cannot open file: " << filename << '\n';
        return false;
    }

    string line;
    getline(file, line);
    while (getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        stringstream ss(line);

        double timeIndex;
        int step, objectIndex;
        char comma1, comma2, comma3, comma4, comma5;
        Position p;

        if (!(ss >> timeIndex >> comma1 >> step >> comma2 >> objectIndex >> comma3 >> p.x >>
              comma4 >> p.y >> comma5 >> p.z))
        {
            cerr << "Invalid CSV line: " << line << '\n';
            return false;
        }

        if (comma1 != ',' || comma2 != ',' || comma3 != ',' || comma4 != ',' || comma5 != ',')
        {
            cerr << "Invalid CSV line: " << line << '\n';
            return false;
        }

        Key key = {timeIndex, step, objectIndex};

        if (positions.count(key) > 0)
        {
            cerr << "Duplicate timeIndex, step, objectIndex: " << timeIndex << ", " << step << ", "
                 << objectIndex << '\n';
            return false;
        }

        positions[key] = p;
    }

    return true;
}

int main()
{
    map<Key, Position> baseline;
    map<Key, Position> optimized;

    if (!ReadCSV("E:/UIT/KLTN/ver3/gravity_sim/numerical_error/baseline_positions.csv", baseline) ||
        !ReadCSV("E:/UIT/KLTN/ver3/gravity_sim/numerical_error/optimized_positions.csv", optimized))
    {
        return 1;
    }

    if (baseline.empty() || optimized.empty())
    {
        cerr << "Invalid CSV data.\n";
        return 1;
    }

    if (baseline.size() != optimized.size())
    {
        cerr << "So luong mau giua hai file khac nhau.\n";
        return 1;
    }

    ofstream out("E:/UIT/KLTN/ver3/gravity_sim/numerical_error/position_error.csv");
    if (!out.is_open())
    {
        cerr << "Cannot create output file.\n";
        return 1;
    }
    out << fixed << setprecision(10);
    out << "time,step,object_id,dx,dy,dz,error,mape\n";
    map<pair<double, int>, pair<double, int>> errorByTime;
    double totalSquaredError = 0.0;
    long long totalObjects = 0;

    std::cout<< fixed << setprecision(10);
    std::cout << "time,step,object_id,dx,dy,dz,error,mape\n";

    for (const auto &entry : baseline)
    {
        const Key &key = entry.first;
        const Position &r = entry.second;

        auto it = optimized.find(key);

        if (it == optimized.end())
        {
            cerr << "Lack optimized data at timeIndex=" << get<0>(key) << ", step=" << get<1>(key)
                 << ", objectIndex=" << get<2>(key) << '\n';
            return 1;
        }

        const Position &c = it->second;

        double dx = r.x - c.x;
        double dy = r.y - c.y;
        double dz = r.z - c.z;

        double squaredError = dx * dx + dy * dy + dz * dz;

        double dist = abs(sqrt(squaredError));
        double mape = (dist / sqrt(r.x * r.x + r.y * r.y + r.z * r.z)) * 100.0;
        out << get<0>(key) << ',' << get<1>(key) << ',' << get<2>(key) << ',' << dx << ',' << dy
            << ',' << dz << ',' << dist << ',' << mape << '\n';

        std::cout << get<0>(key) << ',' << get<1>(key) << ',' << get<2>(key) << ',' << dx << ',' << dy
            << ',' << dz << ',' << dist << ',' << mape << '\n';    

        totalSquaredError += squaredError;
        totalObjects++;

        // Accumulate errors by time
        auto &acc = errorByTime[{get<0>(key), get<1>(key)}];
        acc.first += squaredError;
        acc.second++;
    }

    std::cout << "RMSE tai tung moc thoi gian:\n";

    for (const auto &entry : errorByTime)
    {
        double timeIndex = entry.first.first;
        int step = entry.first.second;
        double sumSquaredError = entry.second.first;
        int count = entry.second.second;

        double rmse = sqrt(sumSquaredError / count);

        std::cout << "timeIndex = " << timeIndex << ", step = " << step << ", RMSE = " << rmse << '\n';
    }

    double globalRMSE = sqrt(totalSquaredError / totalObjects);

    std::cout << "\nSo mau da so sanh: " << totalObjects << '\n';

    std::cout << "RMSE tong the = " << globalRMSE << '\n';

    out.close();

    return 0;
}