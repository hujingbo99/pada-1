#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <unordered_set>
#include <list>
#include <assert.h>

namespace hnswlib
{
    class StopH
    {
        std::chrono::steady_clock::time_point time_begin;

    public:
        StopH()
        {
            time_begin = std::chrono::steady_clock::now();
        }

        float getElapsedTimeMicro()
        {
            std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
            return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
        }

        void reset()
        {
            time_begin = std::chrono::steady_clock::now();
        }
    };

    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;
    template <typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t>
    {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t> *s)
        {
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements = 0)
        {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) : link_list_locks_(max_elements), element_levels_(max_elements), link_list_update_locks_(max_update_element_locks)
        {
            max_elements_ = max_elements;
            has_deletions_ = false;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction, M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            num_layer = 3;  
            data_level0_memory_multi_layer = (char **)malloc(sizeof(char *) * num_layer);
            for (int i = 0; i < num_layer; i++)
            {
                data_level0_memory_multi_layer[i] = (char *)malloc(max_elements_ * size_data_per_element_);
                if (data_level0_memory_multi_layer[i] == nullptr)
                    throw std::runtime_error("Not enough memory");
            }

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        struct CompareByFirst
        {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept
            {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW()
        {

            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++)
            {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);

            for (int i = 0; i < num_layer; i++)
            {
                free(data_level0_memory_multi_layer[i]);
            }
            free(data_level0_memory_multi_layer);

            delete visited_list_pool_;
        }

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;
        size_t num_layer;

        double mult_, revSize_;
        int maxlevel_;

        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;

        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;

        char *data_level0_memory_;
        char **data_level0_memory_multi_layer;
        char **linkLists_;
        std::vector<int> element_levels_;

        size_t data_size_;

        bool has_deletions_;

        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        inline labeltype getExternalLabel(tableint internal_id) const
        {
            labeltype return_label;
            memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline labeltype getExternalLabel(tableint internal_id, char *data_level0_memory_) const
        {
            labeltype return_label;
            memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const
        {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const
        {
            return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline labeltype *getExternalLabeLp(tableint internal_id, char *data_level0_memory_) const
        {
            return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const
        {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        inline char *getDataByInternalId(tableint internal_id, char *data_level0_memory_) const
        {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double reverse_size)
        {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int)r;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer)
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass; //存储已经访问过的元素
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id); //根据dist，向top_candidates队列中按(由大到小)顺序添加ep_id
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound)
                {  
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;  
                 
                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0)
                {
                    data = (int *)get_linklist0(curNodeNum);
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                    //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        dmd_hnsw_searchBaseLayer(tableint ep_id, const void *data_point, int layer, std::vector<int> mapping_id)
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;  
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(mapping_id[ep_id]), dist_func_param_);
                top_candidates.emplace(dist, ep_id);  
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound)
                {  
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;  
                 
                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0)
                {
                    data = (int *)get_linklist0(curNodeNum);
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                    //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(mapping_id[*datal]), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(mapping_id[*(datal + 1)]), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(mapping_id[*(datal + j + 1)]), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(mapping_id[candidate_id]));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(mapping_id[candidateSet.top().second]), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        multi_layer_searchBaseLayer(tableint ep_id, const void *data_point, int layer, char *data_layer0)
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;  
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id, data_layer0))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id, data_layer0), dist_func_param_);
                top_candidates.emplace(dist, ep_id);  
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound)
                {  
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;  
                 
                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0)
                {
                    data = (int *)get_linklist0(curNodeNum, data_layer0);
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                    //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal, data_layer0), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1), data_layer0), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1), data_layer0), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id, data_layer0));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second, data_layer0), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id, data_layer0))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        dmd_hnsw_multi_layer_searchBaseLayer(tableint ep_id, const void *data_point, int layer, char *data_layer0, std::vector<int> mapping_id)
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass; 
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;

            if (!isMarkedDeleted(ep_id, data_layer0))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(mapping_id[ep_id], data_layer0), dist_func_param_);
                top_candidates.emplace(dist, ep_id);  
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound)
                {  
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second; 
                 
                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);
                printf("layer=%d\n", layer);
                int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0)
                {
                    data = (int *)get_linklist0(mapping_id[curNodeNum], data_layer0);
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                    //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(mapping_id[*datal], data_layer0), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(mapping_id[*(datal + 1)], data_layer0), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {

                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(mapping_id[*(datal + j + 1)], data_layer0), _MM_HINT_T0);
#endif

                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    printf("layer1_0=%d\n", layer);
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(mapping_id[candidate_id], data_layer0));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    printf("layer1_1=%d\n", layer);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(mapping_id[candidateSet.top().second], data_layer0), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id, data_layer0))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                    printf("layer1_2=%d\n", layer);
                }
                printf("layer1=%d\n", layer);
            }
            printf("ep_id=%d\n", ep_id);
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        parallel_searchBaseLayer(tableint ep_id, const void *data_point, int layer, int vec_start)
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;  
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);  
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound)
                {  
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;  
                
                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0)
                {
                    data = (int *)get_linklist0(curNodeNum);
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                    //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    //if (visited_array[candidate_id] == visited_array_tag || candidate_id > vec_start)
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;

        template <bool has_deletions, bool collect_metrics = false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty())
            {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound)
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *)data);
                //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if (collect_metrics)
                {
                    metric_hops++;
                    metric_distance_computations += size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag))
                    {

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist)
                        {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                             offsetLevel0_, ///////////
                                         _MM_HINT_T0);      ////////////////////////
#endif

                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        template <bool has_deletions, bool collect_metrics = false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        multi_layer_searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef, char *data_layer0) const
        {
            int num_step = 0;
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id, data_layer0))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id, data_layer0), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty())
            {
                //num_step++;

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound)
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id, data_layer0);
                size_t size = getListCount((linklistsizeint *)data);
                //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if (collect_metrics)
                {
                    metric_hops++;
                    metric_distance_computations += size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_layer0 + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_layer0 + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag))
                    {

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id, data_layer0));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist)
                        {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_layer0 + candidate_set.top().second * size_data_per_element_ +
                                             offsetLevel0_, ///////////
                                         _MM_HINT_T0);      ////////////////////////
#endif

                            if (!has_deletions || !isMarkedDeleted(candidate_id, data_layer0))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        template <bool has_deletions, bool collect_metrics = false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        test_multi_layer_searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef, char *data_layer0, int *step, FILE *fp) const
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id, data_layer0))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id, data_layer0), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty())
            {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound)
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id, data_layer0);
                size_t size = getListCount((linklistsizeint *)data);
                //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if (collect_metrics)
                {
                    metric_hops++;
                    metric_distance_computations += size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_layer0 + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_layer0 + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag))
                    {

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id, data_layer0));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist)
                        {
                            candidate_set.emplace(-dist, candidate_id);
                            (*step)++;
                            fprintf(fp, "step%d: %d\n", *step, dist);
#ifdef USE_SSE
                            _mm_prefetch(data_layer0 + candidate_set.top().second * size_data_per_element_ +
                                             offsetLevel0_, ///////////
                                         _MM_HINT_T0);      ////////////////////////
#endif

                            if (!has_deletions || !isMarkedDeleted(candidate_id, data_layer0))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        void getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
            const size_t M)
        {
            if (top_candidates.size() < M)
            {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0)
            {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size())
            {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list)
                {
                    dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                     getDataByInternalId(curent_pair.second),
                                     dist_func_param_);
                    ;

                    if (curdist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list)
            {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        void dmd_hnsw_getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
            const size_t M, std::vector<int> mapping_id)
        {
            if (top_candidates.size() < M)
            {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0)
            {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size())
            {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list)
                {
                    dist_t curdist =
                        fstdistfunc_(getDataByInternalId(mapping_id[second_pair.second]),
                                     getDataByInternalId(mapping_id[curent_pair.second]),
                                     dist_func_param_);
                    ;

                    if (curdist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list)
            {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        void multi_layer_getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
            const size_t M, char *data_layer0)
        {
            if (top_candidates.size() < M)
            {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0)
            {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size())
            {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list)
                {
                    dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second, data_layer0),
                                     getDataByInternalId(curent_pair.second, data_layer0),
                                     dist_func_param_);
                    ;

                    if (curdist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list)
            {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        void dmd_hnsw_multi_layer_getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
            const size_t M, char *data_layer0, std::vector<int> mapping_id)
        {
            if (top_candidates.size() < M)
            {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0)
            {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size())
            {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list)
                {
                    dist_t curdist =
                        fstdistfunc_(getDataByInternalId(mapping_id[second_pair.second], data_layer0),
                                     getDataByInternalId(mapping_id[curent_pair.second], data_layer0),
                                     dist_func_param_);
                    ;

                    if (curdist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list)
            {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        linklistsizeint *get_linklist0(tableint internal_id) const
        {
            return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0) const
        {
            return (linklistsizeint *)(data_level0 + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist(tableint internal_id, int level) const
        {
            return (linklistsizeint *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const
        {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                           std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                                           int level, bool isUpdate)
        {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            
            getNeighborsByHeuristic2(top_candidates, Mcurmax);  
            if (top_candidates.size() > Mcurmax)                
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

             
            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(Mcurmax);  
            while (top_candidates.size() > 0)
            {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors[0];

            
            {
                linklistsizeint *ll_cur;

                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate)
                {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur, selectedNeighbors.size());
                tableint *data = (tableint *)(ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            

            if (level == 0)
            {

                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {

                    std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                    linklistsizeint *ll_other;
                    if (level == 0)
                        ll_other = get_linklist0(selectedNeighbors[idx]);
                    else
                        ll_other = get_linklist(selectedNeighbors[idx], level);

                    size_t sz_link_list_other = getListCount(ll_other);

                    if (sz_link_list_other > Mcurmax)
                        throw std::runtime_error("Bad value of sz_link_list_other");
                    if (selectedNeighbors[idx] == cur_c)
                        throw std::runtime_error("Trying to connect an element to itself");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    tableint *data = (tableint *)(ll_other + 1);

                    bool is_cur_c_present = false;
                    if (isUpdate)
                    {
                        for (size_t j = 0; j < sz_link_list_other; j++)
                        {
                            if (data[j] == cur_c)
                            {
                                is_cur_c_present = true;
                                break;
                            }
                        }
                    }

                    // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.

                    if (!is_cur_c_present)
                    {

                        if (sz_link_list_other < Mcurmax)
                        {
                            data[sz_link_list_other] = cur_c;
                            setListCount(ll_other, sz_link_list_other + 1);
                        }

                        else
                        //if (sz_link_list_other >= Mcurmax)
                        {
                            //if (sz_link_list_other >= Mcurmax) {
                            // finding the "weakest" element to replace it with the new one
                            dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                        dist_func_param_);
                            // Heuristic:
                            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                            candidates.emplace(d_max, cur_c);

                            for (size_t j = 0; j < sz_link_list_other; j++)
                            {
                                candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_),
                                    data[j]);
                            }

                            getNeighborsByHeuristic2(candidates, Mcurmax);

                            int indx = 0;
                            while (candidates.size() > 0)
                            //while (indx < Mcurmax && candidates.size() > 0)
                            {
                                data[indx] = candidates.top().second;
                                candidates.pop();
                                indx++;
                            }

                            setListCount(ll_other, indx);
                            // Nearest K:
                            //int indx = -1;
                            //for (int j = 0; j < sz_link_list_other; j++) {
                            //dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            //if (d > d_max) {
                            //indx = j;
                            //d_max = d;
                            //}
                            //}
                            //if (indx >= 0) {
                            //data[indx] = cur_c;
                            //}
                        }
                    }
                }
            }

            return next_closest_entry_point;
        }

        tableint dmd_hnsw_mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                                                    int level, bool isUpdate, std::vector<int> mapping_id)
        {
            size_t Mcurmax = level ? maxM_ : maxM0_;
             
            dmd_hnsw_getNeighborsByHeuristic2(top_candidates, Mcurmax, mapping_id); //原始代码为M_
            if (top_candidates.size() > Mcurmax)                                    //原始代码为M_
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

             
            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(Mcurmax); 
            while (top_candidates.size() > 0)
            {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors[0];

            
            {
                linklistsizeint *ll_cur;

                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate)
                {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur, selectedNeighbors.size());
                tableint *data = (tableint *)(ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

             
            //if (level == 0)
            //{

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
            {

                std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *)(ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate)
                {
                    for (size_t j = 0; j < sz_link_list_other; j++)
                    {
                        if (data[j] == cur_c)
                        {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.

                if (!is_cur_c_present)
                {

                    if (sz_link_list_other < Mcurmax)
                    {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    }

                    else
                    //if (sz_link_list_other >= Mcurmax)
                    {
                        //if (sz_link_list_other >= Mcurmax) {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(mapping_id[cur_c]), getDataByInternalId(mapping_id[selectedNeighbors[idx]]),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++)
                        {
                            candidates.emplace(
                                fstdistfunc_(getDataByInternalId(mapping_id[data[j]]), getDataByInternalId(mapping_id[selectedNeighbors[idx]]),
                                             dist_func_param_),
                                data[j]);
                        }

                        dmd_hnsw_getNeighborsByHeuristic2(candidates, Mcurmax, mapping_id);

                        int indx = 0;
                        while (candidates.size() > 0)
                        //while (indx < Mcurmax && candidates.size() > 0)
                        {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        //int indx = -1;
                        //for (int j = 0; j < sz_link_list_other; j++) {
                        //dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        //if (d > d_max) {
                        //indx = j;
                        //d_max = d;
                        //}
                        //}
                        //if (indx >= 0) {
                        //data[indx] = cur_c;
                        //}
                    }
                }
            }

            //}

            return next_closest_entry_point;
        }

        tableint multi_layer_mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                                                       int level, bool isUpdate, char *data_layer0)
        {
            size_t Mcurmax = level ? maxM_ : maxM0_;
             
            multi_layer_getNeighborsByHeuristic2(top_candidates, Mcurmax, data_layer0); //原始代码为M_
            if (top_candidates.size() > Mcurmax)                                        //原始代码为M_
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            
            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(Mcurmax);  
            while (top_candidates.size() > 0)
            {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors[0];

            
            {
                linklistsizeint *ll_cur;

                if (level == 0)
                    ll_cur = get_linklist0(cur_c, data_layer0);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate)
                {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur, selectedNeighbors.size());
                tableint *data = (tableint *)(ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            

            if (level == 0)
            {

                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {

                    std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                    linklistsizeint *ll_other;
                    if (level == 0)
                        ll_other = get_linklist0(selectedNeighbors[idx], data_layer0);
                    else
                        ll_other = get_linklist(selectedNeighbors[idx], level);

                    size_t sz_link_list_other = getListCount(ll_other);

                    if (sz_link_list_other > Mcurmax)
                        throw std::runtime_error("Bad value of sz_link_list_other");
                    if (selectedNeighbors[idx] == cur_c)
                        throw std::runtime_error("Trying to connect an element to itself");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    tableint *data = (tableint *)(ll_other + 1);

                    bool is_cur_c_present = false;
                    if (isUpdate)
                    {
                        for (size_t j = 0; j < sz_link_list_other; j++)
                        {
                            if (data[j] == cur_c)
                            {
                                is_cur_c_present = true;
                                break;
                            }
                        }
                    }

                    // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.

                    if (!is_cur_c_present)
                    {

                        if (sz_link_list_other < Mcurmax)
                        {
                            data[sz_link_list_other] = cur_c;
                            setListCount(ll_other, sz_link_list_other + 1);
                        }

                        else
                        //if (sz_link_list_other >= Mcurmax)
                        {
                            //if (sz_link_list_other >= Mcurmax) {
                            // finding the "weakest" element to replace it with the new one
                            dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c, data_layer0), getDataByInternalId(selectedNeighbors[idx], data_layer0),
                                                        dist_func_param_);
                            // Heuristic:
                            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                            candidates.emplace(d_max, cur_c);

                            for (size_t j = 0; j < sz_link_list_other; j++)
                            {
                                candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j], data_layer0), getDataByInternalId(selectedNeighbors[idx], data_layer0),
                                                 dist_func_param_),
                                    data[j]);
                            }

                            multi_layer_getNeighborsByHeuristic2(candidates, Mcurmax, data_layer0);

                            int indx = 0;
                            while (candidates.size() > 0)
                            //while (indx < Mcurmax && candidates.size() > 0)
                            {
                                data[indx] = candidates.top().second;
                                candidates.pop();
                                indx++;
                            }

                            setListCount(ll_other, indx);
                            // Nearest K:
                            //int indx = -1;
                            //for (int j = 0; j < sz_link_list_other; j++) {
                            //dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            //if (d > d_max) {
                            //indx = j;
                            //d_max = d;
                            //}
                            //}
                            //if (indx >= 0) {
                            //data[indx] = cur_c;
                            //}
                        }
                    }
                }
            }

            return next_closest_entry_point;
        }

        tableint dmd_hnsw_multi_layer_mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                                                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                                                                int level, bool isUpdate, char *data_layer0, std::vector<int> mapping_id)
        {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            
            dmd_hnsw_multi_layer_getNeighborsByHeuristic2(top_candidates, Mcurmax, data_layer0, mapping_id); //原始代码为M_
            if (top_candidates.size() > Mcurmax)                                                             //原始代码为M_
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

             
            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(Mcurmax);  
            while (top_candidates.size() > 0)
            {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors[0];

             
            {
                linklistsizeint *ll_cur;

                if (level == 0)
                    ll_cur = get_linklist0(mapping_id[cur_c], data_layer0);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate)
                {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur, selectedNeighbors.size());
                tableint *data = (tableint *)(ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            

            //if (level == 0)
            //{

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
            {

                std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(mapping_id[selectedNeighbors[idx]], data_layer0);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *)(ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate)
                {
                    for (size_t j = 0; j < sz_link_list_other; j++)
                    {
                        if (data[j] == cur_c)
                        {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.

                if (!is_cur_c_present)
                {

                    if (sz_link_list_other < Mcurmax)
                    {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    }

                    else
                    //if (sz_link_list_other >= Mcurmax)
                    {
                        //if (sz_link_list_other >= Mcurmax) {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(mapping_id[cur_c], data_layer0), getDataByInternalId(mapping_id[selectedNeighbors[idx]], data_layer0),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++)
                        {
                            candidates.emplace(
                                fstdistfunc_(getDataByInternalId(mapping_id[data[j]], data_layer0), getDataByInternalId(mapping_id[selectedNeighbors[idx]], data_layer0),
                                             dist_func_param_),
                                data[j]);
                        }

                        dmd_hnsw_multi_layer_getNeighborsByHeuristic2(candidates, Mcurmax, data_layer0, mapping_id);

                        int indx = 0;
                        while (candidates.size() > 0)
                        //while (indx < Mcurmax && candidates.size() > 0)
                        {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        //int indx = -1;
                        //for (int j = 0; j < sz_link_list_other; j++) {
                        //dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        //if (d > d_max) {
                        //indx = j;
                        //d_max = d;
                        //}
                        //}
                        //if (indx >= 0) {
                        //data[indx] = cur_c;
                        //}
                    }
                }
            }

            //}

            return next_closest_entry_point;
        }

        tableint batch_mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                                 std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                                                 int level, bool isUpdate)
        {
            size_t Mcurmax = level ? maxM_ : maxM0_;
             
            getNeighborsByHeuristic2(top_candidates, Mcurmax);  
            if (top_candidates.size() > Mcurmax)                
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

             
            std::vector<tableint> selectedNeighbors;
            //std::priority_queue<tableint> Neighbors_queue;
            selectedNeighbors.reserve(Mcurmax); 
            while (top_candidates.size() > 0)
            {
                selectedNeighbors.push_back(top_candidates.top().second);
                //Neighbors_queue.emplace(-top_candidates.top().second);
                top_candidates.pop();
            }
            /*
             while (Neighbors_queue.size() > 0)
            {
                selectedNeighbors.push_back(-Neighbors_queue.top());
                Neighbors_queue.pop();
            }
            */
            tableint next_closest_entry_point = selectedNeighbors[0];

             
            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate)
                {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur, selectedNeighbors.size());
                tableint *data = (tableint *)(ll_cur + 1);

                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

          

            return next_closest_entry_point;
        }

        void neighbors_connect(tableint cur_c, tableint selected_N)
        {
            linklistsizeint *ll_other;
            ll_other = get_linklist0(selected_N);
            tableint *data = (tableint *)(ll_other + 1);
            size_t sz_link_list_other = getListCount(ll_other);
            dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selected_N),
                                        dist_func_param_);
            // Heuristic:
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
            candidates.emplace(d_max, cur_c);

            for (size_t j = 0; j < sz_link_list_other; j++)
            {
                candidates.emplace(
                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selected_N),
                                 dist_func_param_),
                    data[j]);
            }

            getNeighborsByHeuristic2(candidates, maxM0_);

            int indx = 0;
            while (candidates.size() > 0)
            //while (indx < Mcurmax && candidates.size() > 0)
            {
                data[indx] = candidates.top().second;
                candidates.pop();
                indx++;
            }

            setListCount(ll_other, indx);
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef)
        {
            ef_ = ef;
        }

        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k)
        {
            std::priority_queue<std::pair<dist_t, tableint>> top_candidates;
            if (cur_element_count == 0)
                return top_candidates;
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (size_t level = maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    int *data;
                    data = (int *)get_linklist(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            if (has_deletions_)
            {
                std::priority_queue<std::pair<dist_t, tableint>> top_candidates1 = searchBaseLayerST<true>(currObj, query_data,
                                                                                                           ef_);
                top_candidates.swap(top_candidates1);
            }
            else
            {
                std::priority_queue<std::pair<dist_t, tableint>> top_candidates1 = searchBaseLayerST<false>(currObj, query_data,
                                                                                                            ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements)
        {
            if (new_max_elements < cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);

            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char *data_level0_memory_new = (char *)malloc(new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            memcpy(data_level0_memory_new, data_level0_memory_, cur_element_count * size_data_per_element_);
            free(data_level0_memory_);
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char **linkLists_new = (char **)malloc(sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            memcpy(linkLists_new, linkLists_, cur_element_count * sizeof(void *));
            free(linkLists_);
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        void saveIndex(const std::string &location)
        {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++)
            {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0)
        {

            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0, input.end);
            std::streampos total_filesize = input.tellg();
            input.seekg(0, input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if (max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos = input.tellg();

            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_, input.cur);
            for (size_t i = 0; i < cur_element_count; i++)
            {
                if (input.tellg() < 0 || input.tellg() >= total_filesize)
                {
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0)
                {
                    input.seekg(linkListSize, input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if (input.tellg() != total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos, input.beg);

            data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++)
            {
                label_lookup_[getExternalLabel(i)] = i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0)
                {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                }
                else
                {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *)malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            has_deletions_ = false;

            for (size_t i = 0; i < cur_element_count; i++)
            {
                if (isMarkedDeleted(i))
                    has_deletions_ = true;
            }

            input.close();

            return;
        }

        template <typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label)
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char *data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *)dist_func_param_);
            std::vector<data_t> data;
            data_t *data_ptr = (data_t *)data_ptrv;
            for (int i = 0; i < dim; i++)
            {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
        //        static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            has_deletions_ = true;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end())
            {
                throw std::runtime_error("Label not found");
            }
            markDeletedInternal(search->second);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId)
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur |= DELETE_MARK;
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId)
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            return *ll_cur & DELETE_MARK;
        }

        bool isMarkedDeleted(tableint internalId, char *data_level0) const
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId, data_level0)) + 2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint *ptr) const
        {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint *ptr, unsigned short int size) const
        {
            *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
        }

        void addPoint(const void *data_point, labeltype label)
        {
            addPoint(data_point, label, -1);
        }

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability)
        {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++)
            {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto &&elOneHop : listOneHop)
                {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto &&elTwoHop : listTwoHop)
                    {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto &&neigh : sNeigh)
                {
                    //                    if (neigh == internalId)
                    //                        continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    int size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;
                    int elementsToKeep = std::min(int(ef_construction_), size);
                    for (auto &&cand : sCand)
                    {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep)
                        {
                            candidates.emplace(distance, cand);
                        }
                        else
                        {
                            if (distance < candidates.top().first)
                            {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        int candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *)(ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++)
                        {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        void repairConnectionsForUpdate(const void *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel)
        {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel)
            {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--)
                {
                    bool changed = true;
                    while (changed)
                    {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj, level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++)
                        {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist)
                            {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--)
            {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0)
                {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0)
                {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted)
                    {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level)
        {
            std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *)(data + 1);
            memcpy(result.data(), ll, size * sizeof(tableint));
            return result;
        };

        tableint addPoint(const void *data_point, labeltype label, int level)
        {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end())
                {
                    tableint existingInternalId = search->second;

                    templock_curr.unlock();

                    std::unique_lock<std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);
                    updatePoint(data_point, existingInternalId, 1.0);
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_)
                {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock<std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0) //level = -1, 不执行
                curlevel = level;

            element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
            for (int i = 0; i < num_layer; i++)
            {
                memset(data_level0_memory_multi_layer[i] + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
            }

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            if (curlevel)
            {
                linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1)
            {

                if (curlevel < maxlevelcopy)
                {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    
                    for (int level = maxlevelcopy; level > curlevel; level--)
                    {
                        bool changed = true;
                        while (changed)
                        {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);
                            
                            tableint *datal = (tableint *)(data + 1);
                            for (int i = 0; i < size; i++)
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--)
                {
                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                    if (epDeleted)
                    {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                     
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }
            }
            else
            {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy)
            {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

         
        tableint multi_layer0_addPoint(const void *data_point, labeltype label, int level, float *down_curlevel, float *other_curlevel)
        {
            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end())
                {
                    tableint existingInternalId = search->second;

                    templock_curr.unlock();

                    std::unique_lock<std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);
                    updatePoint(data_point, existingInternalId, 1.0);
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_)
                {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock<std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            //int curlevel;
            if (level > 0)  
                curlevel = level;

            element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);
             
            for (int i = 0; i < num_layer; i++)
            {
                memset(data_level0_memory_multi_layer[i] + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
                // Initialisation of the data and label
                memcpy(getExternalLabeLp(cur_c, data_level0_memory_multi_layer[i]), &label, sizeof(labeltype));
                memcpy(getDataByInternalId(cur_c, data_level0_memory_multi_layer[i]), data_point, data_size_);
            }

            if (curlevel)
            {
                linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1)
            {
                
                StopH stop_l = StopH();
                float up_curlevel = 0;
                //float down_curlevel = 0;
                //float other_curlevel = 0;
                stop_l.reset();
                if (curlevel < maxlevelcopy)
                {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    
                    for (int level = maxlevelcopy; level > curlevel; level--)
                    {
                        bool changed = true;
                        while (changed)
                        {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);
                            
                            tableint *datal = (tableint *)(data + 1);
                            for (int i = 0; i < size; i++)
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
                up_curlevel = stop_l.getElapsedTimeMicro() / 1e3;
                 
                stop_l.reset();

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level > 0; level--)
                {
                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                    if (epDeleted)
                    {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                     
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }
                //down_curlevel = stop_l.getElapsedTimeMicro() / 1e3;
                 

                stop_l.reset();
                char *data_layer0;
                int level = 0;

                if (curlevel == 0)
                {

                    int i = rand() % (num_layer + 1);

                    if (i == 0)
                        data_layer0 = data_level0_memory_;
                    else
                        data_layer0 = data_level0_memory_multi_layer[i - 1];

                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    stop_l.reset();
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = multi_layer_searchBaseLayer(
                        currObj, data_point, level, data_layer0);  
                    *other_curlevel += stop_l.getElapsedTimeMicro() / 1e3;

                    if (epDeleted)
                    {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy, data_layer0), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }

                    //printf("other_curlevel1 = %f ms\n", other_curlevel);

                    stop_l.reset();
                    currObj = multi_layer_mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false, data_layer0);
                    *down_curlevel += stop_l.getElapsedTimeMicro() / 1e3;

                    // if(label == (1000000-10)){
                    //     printf("other_curlevel1 = %f ms\n", *other_curlevel);
                    //     printf("down_curlevel1 = %f ms\n", *down_curlevel);
                    //     exit(1);
                    // }
                }

                else
                {
                    int vertex;
                    for (int i = 0; i <= num_layer; i++)
                    {
                        vertex = currObj;
                        if (i == 0)
                            data_layer0 = data_level0_memory_;
                        else
                            data_layer0 = data_level0_memory_multi_layer[i - 1];

                        if (level > maxlevelcopy || level < 0) // possible?
                            throw std::runtime_error("Level error");

                        stop_l.reset();
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = multi_layer_searchBaseLayer(
                            vertex, data_point, level, data_layer0);
                        *other_curlevel += stop_l.getElapsedTimeMicro() / 1e3;

                        if (epDeleted)
                        {
                            top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy, data_layer0), dist_func_param_), enterpoint_copy);
                            if (top_candidates.size() > ef_construction_)
                                top_candidates.pop();
                        }

                        //other_curlevel = stop_l.getElapsedTimeMicro() / 1e3;
                        //printf("other_curlevel = %f ms\n", other_curlevel);

                        stop_l.reset();
                        vertex = multi_layer_mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false, data_layer0);
                        *down_curlevel += stop_l.getElapsedTimeMicro() / 1e3;
                        //printf("down_curlevel = %f ms\n", down_curlevel);
                        //exit(1);
                    }
                }

                //other_curlevel = stop_l.getElapsedTimeMicro() / 1e3;
                //printf("other_curlevel = %f ms\n", other_curlevel);
                //exit(1);
                //printf("2_1\n");
            }
            else
            {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }
           
            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy)
            {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

         
        tableint multi_layer0_addPoint_memory(const void *data_point, labeltype label, int level, float *down_curlevel, float *other_curlevel, std::vector<int> mapping_layer, std::vector<int> mapping_id)
        {
            tableint cur_c = 0;
            printf("label = %d\n", label);
            printf("mapping_id = %d\n", mapping_id[label]);
            printf("mapping_layer = %d\n", mapping_layer[label]);
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.

                std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);

                if (search != label_lookup_.end())
                {
                    tableint existingInternalId = search->second;

                    templock_curr.unlock();

                    std::unique_lock<std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);
                    updatePoint(data_point, existingInternalId, 1.0);
                    return existingInternalId;
                }
                //printf("h");
                if (cur_element_count >= max_elements_)
                {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }
            
            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock<std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            //int curlevel = getRandomLevel(mult_);
            int curlevel;
            if (level >= 0)  
                curlevel = level;

            element_levels_[cur_c] = curlevel;
            printf("level = %d\n", curlevel);

            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            if (curlevel)
            {
                printf("1\n");
                linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);

                memset(data_level0_memory_ + mapping_id[cur_c] * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
                // Initialisation of the data and label
                memcpy(getExternalLabeLp(mapping_id[cur_c]), &label, sizeof(labeltype));
                memcpy(getDataByInternalId(mapping_id[cur_c]), data_point, data_size_);
                 
                for (int i = 0; i < num_layer; i++)
                {
                    memset(data_level0_memory_multi_layer[i] + mapping_id[cur_c] * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
                    // Initialisation of the data and label
                    memcpy(getExternalLabeLp(mapping_id[cur_c], data_level0_memory_multi_layer[i]), &label, sizeof(labeltype));
                    memcpy(getDataByInternalId(mapping_id[cur_c], data_level0_memory_multi_layer[i]), data_point, data_size_);
                }
            }
            else
            {

                if (mapping_layer[cur_c] == 0)
                {
                    memset(data_level0_memory_ + mapping_id[cur_c] * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
                    // Initialisation of the data and label
                    memcpy(getExternalLabeLp(mapping_id[cur_c]), &label, sizeof(labeltype));
                    memcpy(getDataByInternalId(mapping_id[cur_c]), data_point, data_size_);
                }
                else
                {

                    memset(data_level0_memory_multi_layer[mapping_layer[cur_c] - 1] + mapping_id[cur_c] * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

                    // Initialisation of the data and label
                    memcpy(getExternalLabeLp(mapping_id[cur_c], data_level0_memory_multi_layer[mapping_layer[cur_c] - 1]), &label, sizeof(labeltype));
                    memcpy(getDataByInternalId(mapping_id[cur_c], data_level0_memory_multi_layer[mapping_layer[cur_c] - 1]), data_point, data_size_);
                }
            }

            if ((signed)currObj != -1)
            {
                // 3.16 Hu test
                StopH stop_l = StopH();
                float up_curlevel = 0;
                //float down_curlevel = 0;
                //float other_curlevel = 0;
                stop_l.reset();
                if (curlevel < maxlevelcopy)
                {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(mapping_id[currObj]), dist_func_param_);
                    
                    for (int level = maxlevelcopy; level > curlevel; level--)
                    {
                        bool changed = true;
                        while (changed)
                        {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);
                           
                            tableint *datal = (tableint *)(data + 1);
                            for (int i = 0; i < size; i++)
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(mapping_id[cand]), dist_func_param_);
                                if (d < curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                up_curlevel = stop_l.getElapsedTimeMicro() / 1e3;
                 
                stop_l.reset();

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level > 0; level--)
                {
                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = dmd_hnsw_searchBaseLayer(
                        currObj, data_point, level, mapping_id);
                    if (epDeleted)
                    {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(mapping_id[enterpoint_copy]), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    
                    currObj = dmd_hnsw_mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false, mapping_id);
                }

                //down_curlevel = stop_l.getElapsedTimeMicro() / 1e3;
                //printf("down_curlevel = %f ms\n", down_curlevel);

                stop_l.reset();
                char *data_layer0;
                int level = 0;

                if (curlevel == 0)
                {
                    
                    int i = mapping_layer[cur_c];

                    if (i == 0)
                        data_layer0 = data_level0_memory_;
                    else
                        data_layer0 = data_level0_memory_multi_layer[i - 1];

                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    stop_l.reset();
                    printf("level1 = %d\n", curlevel);
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = dmd_hnsw_multi_layer_searchBaseLayer(
                        currObj, data_point, level, data_layer0, mapping_id);  
                    printf("level2 = %d\n", curlevel);
                    *other_curlevel += stop_l.getElapsedTimeMicro() / 1e3;

                    if (epDeleted)
                    {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(mapping_id[enterpoint_copy], data_layer0), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }

                    

                    stop_l.reset();
                    currObj = dmd_hnsw_multi_layer_mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false, data_layer0, mapping_id);
                    *down_curlevel += stop_l.getElapsedTimeMicro() / 1e3;
                    printf("level = %d\n", curlevel);
                    
                }

                else
                {
                    int vertex;
                    for (int i = 0; i <= num_layer; i++)
                    {
                        vertex = currObj;
                        if (i == 0)
                            data_layer0 = data_level0_memory_;
                        else
                            data_layer0 = data_level0_memory_multi_layer[i - 1];

                        if (level > maxlevelcopy || level < 0) // possible?
                            throw std::runtime_error("Level error");

                        stop_l.reset();
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = dmd_hnsw_multi_layer_searchBaseLayer(
                            vertex, data_point, level, data_layer0, mapping_id);
                        *other_curlevel += stop_l.getElapsedTimeMicro() / 1e3;

                        if (epDeleted)
                        {
                            top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(mapping_id[enterpoint_copy], data_layer0), dist_func_param_), enterpoint_copy);
                            if (top_candidates.size() > ef_construction_)
                                top_candidates.pop();
                        }


                        stop_l.reset();
                        vertex = dmd_hnsw_multi_layer_mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false, data_layer0, mapping_id);
                        *down_curlevel += stop_l.getElapsedTimeMicro() / 1e3;
                        
                    }
                     
                }

            }
            else
            {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }
             
            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy)
            {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        tableint parallel_addPoint(const void *data_point, labeltype label, int level, int vec_start)
        {
            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end())
                {
                    tableint existingInternalId = search->second;

                    templock_curr.unlock();

                    std::unique_lock<std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);
                    updatePoint(data_point, existingInternalId, 1.0);
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_)
                {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock<std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0) //level = -1, 不执行
                curlevel = level;

            element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            if (curlevel)
            {
                linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1)
            {

                if (curlevel < maxlevelcopy)
                {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    
                    for (int level = maxlevelcopy; level > curlevel; level--)
                    {
                        bool changed = true;
                        while (changed)
                        {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);
                             
                            tableint *datal = (tableint *)(data + 1);
                            for (int i = 0; i < size; i++)
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                //if (cand < vec_start)
                                //{
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                                //}
                            }
                        }
                    }
                }

                
                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--)
                {
                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = parallel_searchBaseLayer(
                        currObj, data_point, level, vec_start);
                    if (epDeleted)
                    {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                     
                    currObj = batch_mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }
            }
            else
            {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy)
            {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void *query_data, size_t k) const
        {
            
            std::priority_queue<std::pair<dist_t, labeltype>> result;
            if (cur_element_count == 0)
                return result;

            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *)get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

             
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> multi_layer_top_candidates(num_layer + 1);
            std::vector<int> element_flag = std::vector<int>(max_elements_);
#pragma omp parallel for num_threads(3)
            for (int i = 0; i <= num_layer; i++)
            {
                //printf("2");
                //int i = rand() % 3;
                char *data_layer0;
                //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> multi_layer_top_candidates;
                if (i == 0)
                    data_layer0 = data_level0_memory_;
                else
                    data_layer0 = data_level0_memory_multi_layer[i - 1];

                if (has_deletions_)
                {
                    //top_candidates = searchBaseLayerST<true, true>(
                    //currObj, query_data, std::max(ef_, k));

                    multi_layer_top_candidates[i] = multi_layer_searchBaseLayerST<true, true>(
                        currObj, query_data, std::max(ef_, k), data_layer0);
                }

                else
                {
                    //top_candidates = searchBaseLayerST<false, true>(
                    //currObj, query_data, std::max(ef_, k));

                    multi_layer_top_candidates[i] = multi_layer_searchBaseLayerST<false, true>(
                        currObj, query_data, std::max(ef_, k), data_layer0);
                }
            }

            for (int i = 0; i <= num_layer; i++)
            {
                while (multi_layer_top_candidates[i].size() > 0)
                {
                    if (element_flag[multi_layer_top_candidates[i].top().second] != 1)
                    {
                        top_candidates.emplace(multi_layer_top_candidates[i].top().first, multi_layer_top_candidates[i].top().second);
                        element_flag[multi_layer_top_candidates[i].top().second] = 1;
                        if (top_candidates.size() > k)
                        {
                            top_candidates.pop();
                        }
                    }
                    multi_layer_top_candidates[i].pop();
                }
            }

            while (top_candidates.size() > 0)
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }

            return result;
        };

        std::priority_queue<std::pair<dist_t, labeltype>>
        test_searchKnn(const void *query_data, size_t k) const
        {
            int x = 0;
            int *step = &x;

            FILE *fp = NULL;
            fp = fopen("test.txt", "w+");
            fprintf(fp, "This is a test!\n");

            std::priority_queue<std::pair<dist_t, labeltype>> result;
            if (cur_element_count == 0)
                return result;

            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
            (*step)++;
            fprintf(fp, "step%d: %d\n", *step, curdist);

            for (int level = maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *)get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            (*step)++;
                            fprintf(fp, "step%d: %d\n", *step, curdist);
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

             
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> multi_layer_top_candidates(num_layer + 1);

            std::vector<int> element_flag = std::vector<int>(max_elements_);
#pragma omp parallel for num_threads(num_layer + 1)
            for (int i = 0; i <= 0; i++)
            {
                char *data_layer0;
                if (i == 0)
                    data_layer0 = data_level0_memory_;
                else
                    data_layer0 = data_level0_memory_multi_layer[i - 1];

                if (has_deletions_)
                {
                     
                    multi_layer_top_candidates[i] = test_multi_layer_searchBaseLayerST<true, true>(
                        currObj, query_data, std::max(ef_, k), data_layer0, step, fp);
                }

                else
                {
                    multi_layer_top_candidates[i] = test_multi_layer_searchBaseLayerST<true, true>(
                        currObj, query_data, std::max(ef_, k), data_layer0, step, fp);
                }
            }

            for (int i = 0; i <= num_layer; i++)
            {
                while (multi_layer_top_candidates[i].size() > 0)
                {
                    //if (element_flag[multi_layer_top_candidates[i].top().second] != 1)
                    //{
                    top_candidates.emplace(multi_layer_top_candidates[i].top().first, multi_layer_top_candidates[i].top().second);
                    element_flag[multi_layer_top_candidates[i].top().second] = 1;
                    if (top_candidates.size() > k)
                    {
                        top_candidates.pop();
                    }
                    //}
                    multi_layer_top_candidates[i].pop();
                }
            }


            while (top_candidates.size() > 0)
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            fclose(fp);
            return result;
        };

        template <typename Comp>
        std::vector<std::pair<dist_t, labeltype>>
        searchKnn(const void *query_data, size_t k, Comp comp)
        {
            std::vector<std::pair<dist_t, labeltype>> result;
            if (cur_element_count == 0)
                return result;

            auto ret = searchKnn(query_data, k);

            while (!ret.empty())
            {
                result.push_back(ret.top());
                ret.pop();
            }

            std::sort(result.begin(), result.end(), comp);

            return result;
        }

        void checkIntegrity()
        {
            int connections_checked = 0;
            std::vector<int> inbound_connections_num(cur_element_count, 0);
            for (int i = 0; i < cur_element_count; i++)
            {
                for (int l = 0; l <= element_levels_[i]; l++)
                {
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *)(ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j = 0; j < size; j++)
                    {
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert(data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;
                    }
                    assert(s.size() == size);
                }
            }
            if (cur_element_count > 1)
            {
                int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
                for (int i = 0; i < cur_element_count; i++)
                {
                    assert(inbound_connections_num[i] > 0);
                    min1 = std::min(inbound_connections_num[i], min1);
                    max1 = std::max(inbound_connections_num[i], max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";
        }
    };

}  
