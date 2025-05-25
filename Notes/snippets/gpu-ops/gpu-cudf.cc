*** Intro


// Read CSV
std::string filename = "/home/devavret/Development/1brc/data/measurements.txt";
cudf::io::csv_reader_options in_opts = 
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filename})
      .delimiter(';')
      .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::STRING},
                                             cudf::data_type{cudf::type_id::FLOAT32}})
      .na_filter(false);
auto result = cudf::io::read_csv(in_opts);

// Groupby and Aggregate
auto keys = result.tbl->select({0});
auto groupby_obj = cudf::groupby::groupby(keys);
std::vector<cudf::groupby::aggregation_request> aggregation_reqs{};
auto& req = aggregation_reqs.emplace_back();
req.values = result.tbl->get_column(1);
req.aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
req.aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
req.aggregations.emplace_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
groupby_obj.aggregate(aggregation_reqs);


*** table_view and table

data buffer/ nullmask buffer
table_view支持选列

*** Note

- cuDF cannot handle > 2^31 chars.    ---> 不能GPU把数据一次全弄进一个data frame