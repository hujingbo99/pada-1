#define _GNU_SOURCE
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"
#include "cmath"
#include <unordered_set>
#include <atomic>
#include <thread>

#include <sched.h>

#include <unordered_set>

using namespace std;
using namespace hnswlib;

enum Dataset
{
    sift,
    gist,
    deep,
    glove,
    crawl
};
Dataset using_dataset = sift;
//sift
typedef unsigned char Data_type_set;
typedef int Data_type_NSW;

//gist,deep,glove,crawl
// typedef float Data_type_set;
// typedef float Data_type_NSW;

class StopW
{
    std::chrono::steady_clock::time_point time_begin;

public:
    StopW()
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

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L; /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L; /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L; /* Unsupported. */
#endif
}

/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L; /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t)0L; /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1)
    {
        fclose(fp);
        return (size_t)0L; /* Can't read? */
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L; /* Unsupported. */
#endif
}

static void
get_gt(unsigned int *massQA, Data_type_set *massQ, Data_type_set *mass, size_t vecsize, size_t qsize,
       size_t vecdim, vector<std::priority_queue<std::pair<Data_type_NSW, labeltype>>> &answers, size_t k, int gt_maxnum)
{
    (vector<std::priority_queue<std::pair<Data_type_NSW, labeltype>>>(qsize)).swap(answers);
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++)
    {
        for (int j = 0; j < k; j++)
        {
            answers[i].emplace(0.0f, massQA[gt_maxnum * i + j]);
        }
    }
}

static float
test_approx(Data_type_set *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<Data_type_NSW> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<Data_type_NSW, labeltype>>> &answers, size_t k)
{
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:

//#pragma omp parallel for num_threads(3)
    for (int i = 0; i < qsize; i++)
    {
        std::priority_queue<std::pair<Data_type_NSW, labeltype>> result;

        // if (i == 0)
        // {
        //     //printf("1\n");
        //     result = appr_alg.test_searchKnn(massQ + vecdim * i, k);
        // }

        // else
        // {
            result = appr_alg.searchKnn(massQ + vecdim * i, k);
        // }
        std::priority_queue<std::pair<Data_type_NSW, labeltype>> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size())
        {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size())
        {
            if (g.find(result.top().second) != g.end())
            {
                correct++;
            }
            else
            {
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

static void
test_vs_recall(Data_type_set *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<Data_type_NSW> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<Data_type_NSW, labeltype>>> &answers, size_t k)
{
    vector<size_t> efs; // = { 10,10,10,10,10 };
    
    for (int i = k; i < 30; i++)
    {
        efs.push_back(i);
    }
    
    for (int i = 30; i < 100; i += 10)
    {
        efs.push_back(i);
    }
    
    // for (int i = 100; i < 1000; i += 100)
    // {
    //     efs.push_back(i);
    // }
    // for (int i = 3000; i < 8000; i += 500)
    // {
    //     efs.push_back(i);
    // }

    //efs.push_back(20);
     
    for (size_t ef : efs)
    {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
         
        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
    
        if (recall > 1.0)
        {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
     
}

inline bool exists_test(const std::string &name)
{
    ifstream f(name.c_str());
    return f.good();
}

void sift_test1B()
{
    // cpu_set_t mask;
    // CPU_ZERO(&mask);
    // //CPU_SET(0x0007, &mask);
    // CPU_SET(1, &mask);
    // //int result1 = sched_setaffinity(0, sizeof(mask), &mask);
    // if (sched_setaffinity(0, sizeof(mask), &mask) == -1)
    // {
    //     fprintf(stderr, "warning: could not set CPU affinity/n");
    // }
    // 1M test
     
    int subset_size_milllions = 1;
    int efConstruction = 200;
    int M = 16;

    size_t vecsize = subset_size_milllions * 1000000;
    size_t qsize;
    size_t vecdim;
    size_t gt_maxnum;
    char path_index[1024];
    char path_gt[1024];
    char path_q[1024];
    char path_data[1024];

    if (using_dataset == sift)
    {
        qsize = 10000;
        vecdim = 128;
        gt_maxnum = 1000;
        sprintf(path_q, "bigann/bigann_query.bvecs");
        sprintf(path_data, "bigann/bigann_base.bvecs");
        sprintf(path_index, "sift%dm_ef_%d_M_%d.bin", subset_size_milllions, efConstruction, M);
        sprintf(path_gt, "bigann/gnd/idx_%dM.ivecs", subset_size_milllions);
    }
    else if (using_dataset == gist)
    {
        if (subset_size_milllions > 1)
        {
            printf("error: gist size set error.\n");
            exit(1);
        }
        qsize = 1000;
        vecdim = 960;
        gt_maxnum = 100;
        sprintf(path_q, "gist/gist_query.fvecs");
        sprintf(path_data, "gist/gist_base.fvecs");
        sprintf(path_index, "gist%dm_ef_%d_M_%d.bin", subset_size_milllions, efConstruction, M);
        sprintf(path_gt, "gist/gist_groundtruth.ivecs");
    }
    else if (using_dataset == deep)
    {
        if (subset_size_milllions > 100)
        {
            printf("error: deep size set error.\n");
            exit(1);
        }
        qsize = 10000;
        vecdim = 96;
        gt_maxnum = 100;
        sprintf(path_q, "deep1B/deep1B_queries.fvecs");
        sprintf(path_data, "deep1B/deep_base/deep_base.fvecs");
        sprintf(path_index, "deep%dm_ef_%d_M_%d.bin", subset_size_milllions, efConstruction, M);
        sprintf(path_gt, "deep1B/deep_gnd/idx_%dM.ivecs", subset_size_milllions);
    }
    else if (using_dataset == glove)
    {
        if (subset_size_milllions > 1)
        {
            printf("error: glove size set error.\n");
            exit(1);
        }
        // 1193515 1193517
        vecsize = 1193517;
        qsize = 10000;
        // (25) 50 100 200
        vecdim = 100;
        gt_maxnum = 100;
        sprintf(path_q, "/home/hujingbo99/hnswlib/glove/glove%lud_query.fvecs", vecdim);
        sprintf(path_data, "/home/hujingbo99/hnswlib/glove/glove_base/glove%lud_base.fvecs", vecdim);
        sprintf(path_index, "glove%dm_%lud_ef_%d_M_%d.bin", subset_size_milllions, vecdim, efConstruction, M);
        sprintf(path_gt, "/home/hujingbo99/hnswlib/glove/gnd/idx_%lud.ivecs", vecdim);
    }
    else if (using_dataset == crawl)
    {
        if (subset_size_milllions > 2)
        {
            printf("error: glove size set error.\n");
            exit(1);
        }
        // 42 840
        int tokens = 42;
        if (tokens == 42)
        {
            vecsize = 1917495;
        }
        else if (tokens == 840)
        {
            vecsize = 2196018;
        }
        qsize = 10000;
        vecdim = 300;
        gt_maxnum = 100;
        sprintf(path_q, "/home/hujingbo99/hnswlib/crawl/crawl%dt_query.fvecs", tokens);
        sprintf(path_data, "/home/hujingbo99/hnswlib/crawl/crawl_base/crawl%dt_base.fvecs", tokens);
        sprintf(path_index, "crawl%dm_%dt_ef_%d_M_%d.bin", subset_size_milllions, tokens, efConstruction, M);
        sprintf(path_gt, "/home/hujingbo99/hnswlib/crawl/gnd/idx_%dt.ivecs", tokens);
    }

    L2SpaceI distspace(vecdim); //sift
    //L2Space distspace(vecdim);    //gist
    //InnerProductSpace distspace(vecdim); //deep,glove,crawl

    Data_type_set *massb = new Data_type_set[vecdim];

    cout << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * gt_maxnum];
    for (int i = 0; i < qsize; i++)
    {
        int t;
        inputGT.read((char *)&t, 4);
        inputGT.read((char *)(massQA + gt_maxnum * i), gt_maxnum * 4);
        // if (using_dataset == sift || using_dataset == gist){
        if (t != gt_maxnum)
        {
            cout << "err";
            return;
        }
    }
    inputGT.close();

    cout << "Loading queries:\n";
    Data_type_set *massQ = new Data_type_set[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);

    for (int i = 0; i < qsize; i++)
    {
        if (using_dataset == sift || using_dataset == gist || using_dataset == deep)
        {
            int in = 0;
            inputQ.read((char *)&in, 4);
            if (in != vecdim)
            {
                cout << "file error";
                exit(1);
            }
        }
         
        inputQ.read((char *)massb, vecdim * sizeof(Data_type_set));
        for (int j = 0; j < vecdim; j++)
        {
            massQ[i * vecdim + j] = massb[j];
        }
    }
    inputQ.close();

    Data_type_set *mass = new Data_type_set[vecdim];
    ifstream input(path_data, ios::binary);
    int in = 0;

    HierarchicalNSW<Data_type_NSW> *appr_alg;
    if (exists_test(path_index))
    {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<Data_type_NSW>(&distspace, path_index, false);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    }
    else
    {
        cout << "Building index:\n";
        appr_alg = new HierarchicalNSW<Data_type_NSW>(&distspace, vecsize, M, efConstruction);
        if (using_dataset == sift || using_dataset == gist || using_dataset == deep)
        {
            input.read((char *)&in, 4);
            if (in != vecdim)
            {
                cout << "file error";
                exit(1);
            }
        }
        input.read((char *)massb, vecdim * sizeof(Data_type_set));

        for (int j = 0; j < vecdim; j++)
        {
            mass[j] = massb[j];
        }

        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 100000;
        int original_vecsize = vecsize;
        int batch_size = 10000;
        int batch_index;
        int partition_batch = 10000;


        appr_alg->addPoint((void *)(massb), (size_t)0);  

        StopW stopx = StopW();
        float t_visit_ = 0;
        float *down_curlevel;
        float *other_curlevel;
        down_curlevel = (float *)malloc(sizeof(float));
        other_curlevel = (float *)malloc(sizeof(float));
        *down_curlevel = 0;
        *other_curlevel = 0;

#pragma omp parallel for
        for (int i = 1; i < original_vecsize; i++)
        {
            Data_type_set mass[vecdim];
            int j2 = 0;
            int level = -1;
#pragma omp critical
            {
                if (using_dataset == sift || using_dataset == gist || using_dataset == deep)
                {
                    input.read((char *)&in, 4);
                    if (in != vecdim)
                    {
                        cout << "file error";
                        exit(1);
                    }
                }
                input.read((char *)massb, vecdim * sizeof(Data_type_set));

                for (int j = 0; j < vecdim; j++)
                {
                    mass[j] = massb[j];
  
                }

                j1++;
                j2 = j1;
                if (j1 % report_every == 0)
                {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips "
                         << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
 
            appr_alg->multi_layer0_addPoint((void *)(mass), (size_t)j2, level, down_curlevel, other_curlevel);
        }

        for (batch_index = 0; batch_index < (vecsize - original_vecsize) / batch_size; batch_index++)
        {
            int vec_start = original_vecsize + batch_index * batch_size;
            //int level = appr_alg->getRandomLevel(M);
            int level = -1;
            int limit;
            if (vec_start + batch_size < vecsize)
                limit = vec_start + batch_size;
            else
                limit = vecsize;
            std::priority_queue<std::pair<tableint, tableint>> N_queue;
            std::vector<tableint> node_list;
            std::vector<tableint> N_list;

#pragma omp parallel for
            for (int i = vec_start; i < limit; i++)
            {
                Data_type_set mass[vecdim];
                int j2 = 0;
#pragma omp critical
                {
                    if (using_dataset == sift || using_dataset == gist || using_dataset == deep)
                    {
                        input.read((char *)&in, 4);
                        if (in != vecdim)
                        {
                            cout << "file error";
                            exit(1);
                        }
                    }
                    input.read((char *)massb, vecdim * sizeof(Data_type_set));

                    for (int j = 0; j < vecdim; j++)
                    {
                        mass[j] = massb[j];
                        //printf("mass[j] = %f\n", mass[j]);
                        //exit(1);
                    }

                    j1++;
                    j2 = j1;
                    if (j1 % report_every == 0)
                    {
                        cout << j1 / (0.01 * vecsize) << " %, "
                             << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips "
                             << " Mem: "
                             << getCurrentRSS() / 1000000 << " Mb \n";
                        stopw.reset();
                    }
                }
                 
                appr_alg->multi_layer0_addPoint((void *)(mass), (size_t)j2, level, down_curlevel, other_curlevel);
                 
#pragma omp critical
                {
                    linklistsizeint *ll_other;
                    ll_other = appr_alg->get_linklist0(j2);
                    size_t sz_link_list_other = appr_alg->getListCount(ll_other);
                    tableint *data = (tableint *)(ll_other + 1);
                    for (size_t idx = 0; idx < sz_link_list_other; idx++)
                    {
                        N_queue.emplace(-data[idx], j2);
                    }
                }
            }
             
            //std::priority_queue<tableint> N_queue;
            /*
            std::priority_queue<std::pair<tableint, tableint>> N_queue;
            std::vector<tableint> node_list;
            std::vector<tableint> N_list;
            for (int i = vec_start; i < limit; i++)
            {
                linklistsizeint *ll_other;
                ll_other = appr_alg->get_linklist0(i);
                size_t sz_link_list_other = appr_alg->getListCount(ll_other);
                tableint *data = (tableint *)(ll_other + 1);
                for (size_t idx = 0; idx < sz_link_list_other; idx++)
                {
                    N_queue.emplace(-data[idx], i);
                }
            }
            */

            stopx.reset();
            int count = 1;
            while (N_queue.size() > 0)
            {
                N_list.push_back(-N_queue.top().first);
                node_list.push_back(N_queue.top().second);
                N_queue.pop();
            }

#pragma omp parallel for
            for (size_t idx = 0; idx < N_list.size(); idx++)
            {

                //for (int k = idx; k < idx + 1000; k++)
                //for (int k = vec_start; k < limit; k++)
                //{
                //if( data[idx] > i + partition_batch) break;
                //if ( data[idx] < i + partition_batch && data[idx] >= i)
                //{
                //std::unique_lock<std::mutex> lock(appr_alg->link_list_locks_[N_list[idx]]);
                //std::unique_lock<std::mutex> lock(appr_alg->link_list_locks_[k-10000+rand()%10]);
                //linklistsizeint *ll_other;
                //ll_other = appr_alg->get_linklist0(k);
                //size_t sz_link_list_other = appr_alg->getListCount(ll_other);
                //if (sz_link_list_other >= idx)
                //{
                //tableint *data = (tableint *)(ll_other + 1);
                linklistsizeint *n_ll_other;
                //n_ll_other = appr_alg->get_linklist0(data[idx]);
                n_ll_other = appr_alg->get_linklist0(N_list[idx]);
                //printf("N_list[%d] = %d\n", idx, N_list[idx]);
                size_t n_sz_link_list_other = appr_alg->getListCount(n_ll_other);
                tableint *n_data = (tableint *)(n_ll_other + 1);

                if (n_sz_link_list_other < 2 * M)
                {
                    //n_data[n_sz_link_list_other] = k;
                    n_data[n_sz_link_list_other] = node_list[idx];
                    appr_alg->setListCount(n_ll_other, n_sz_link_list_other + 1);
                }
                else
                {
                    appr_alg->neighbors_connect(node_list[idx], N_list[idx]);
                }

                //}
                //}
                //}
            }

            t_visit_ += stopx.getElapsedTimeMicro() / 1e3;

            //}
        }
        input.close();
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
        appr_alg->saveIndex(path_index);
        printf("The visit used time is %.2f ms.\n", t_visit_);
        printf("-------------------------------------------------------------\n");
    }

    vector<std::priority_queue<std::pair<Data_type_NSW, labeltype>>> answers;
    // recall@1
    size_t k = 1;
    cout << "Parsing gt:\n";
    get_gt(massQA, massQ, mass, vecsize, qsize, vecdim, answers, k, gt_maxnum);
    cout << "Loaded gt\n";
    for (int i = 0; i < 1; i++)
        printf("start\n");
    test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    return;
}
