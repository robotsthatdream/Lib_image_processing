#ifndef TRAINING_DATA_HPP
#define TRAINING_DATA_HPP

#include <iostream>
#include <vector>
#include <memory>


template <typename Data>
class TrainingData{

public:

    typedef std::shared_ptr<TrainingData> Ptr;
    typedef const std::shared_ptr<TrainingData> ConstPtr;

    typedef std::pair<bool,Data> element_t;
    typedef std::vector<element_t> data_t;

    /**
     * @brief default constructor
     */
    TrainingData(){}

    /**
     * @brief copy constructor
     * @param d
     */
    TrainingData(const TrainingData& d) : _data(d._data){}

    /**
     * @brief add an element in the data
     * @param positive data or negative data
     * @param data
     */
    void add(bool label,Data d){
        _data.push_back(std::make_pair(label,d));
        if(label)
            _pos_data.push_back(d);
        else _neg_data.push_back(d);
    }

    /**
     * @brief erase all data
     */
    void clear(){
        _data.clear();
        _neg_data.clear();
        _pos_data.clear();
    }

    /**
     * @brief operator []
     * @param i
     * @return
     */
    const element_t& operator [](size_t i) const {return _data[i];}

    /**
     * @brief operator []
     * @param i
     * @return
     */
    element_t& operator [](size_t i){return _data[i];}

    /**
     * @brief number of elements
     * @return
     */
    size_t size() const {return _data.size();}

    /**
     * @brief get all data
     * @return vector of paired label and data
     */
    const data_t& get(){return _data;}

    /**
     * @brief get only positive data
     * @return vector of positive data
     */
    const std::vector<Data>& get_pos_data(){return _pos_data;}

    /**
     * @brief get only negative data
     * @return vector of negative data
     */
    const std::vector<Data>& get_neg_data(){return _neg_data;}

protected:
    data_t _data;
    std::vector<Data> _neg_data;
    std::vector<Data> _pos_data;
};

#endif //TRAINING_DATA_HPP
