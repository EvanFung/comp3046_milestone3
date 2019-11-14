//
// Created by ManHo Fung on 2019/11/11.
//
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace boost::numeric;
#ifndef COMP3046_MILESTONE_3_DATA_LOADER_H
#define COMP3046_MILESTONE_3_DATA_LOADER_H

template<typename T>
class data_loader {
public:
    data_loader(const std::string &FileData, const std::string &FileLabels,
                std::vector<std::pair<ublas::vector<T>, ublas::vector<T>>> &data) {

    };
};


#endif //COMP3046_MILESTONE_3_DATA_LOADER_H
