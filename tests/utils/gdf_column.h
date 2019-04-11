#ifndef BLAZINGDB_COMMUNICATION_TEST_GDF_COLUMN_H
#define BLAZINGDB_COMMUNICATION_TEST_GDF_COLUMN_H

#include <memory>
#include <cstring>

namespace blazingdb {
namespace test {

    typedef int gdf_size_type;
    typedef unsigned char gdf_valid_type;

    enum gdf_dtype {
        GDF_invalid=0,
        GDF_INT8,
        GDF_INT16,
        GDF_INT32,
        GDF_INT64,
        GDF_FLOAT32,
        GDF_FLOAT64,
        N_GDF_TYPES,
    };

    enum gdf_time_unit {
        TIME_UNIT_NONE=0,
        TIME_UNIT_s,
        TIME_UNIT_ms,
        TIME_UNIT_us,
        TIME_UNIT_ns
    };

    struct gdf_dtype_extra_info {
        gdf_time_unit time_unit;
    };

    struct gdf_column {
        char*                 data;
        gdf_valid_type*       valid;
        gdf_size_type         size;
        gdf_dtype             dtype;
        gdf_size_type         null_count;
        gdf_dtype_extra_info  dtype_info;
        char*                 col_name;
    };

    std::size_t get_dtype_width(blazingdb::test::gdf_dtype type);

    gdf_column* clone(gdf_column* input_column);

    gdf_column* make(char* input_data,
                     gdf_valid_type* input_valid,
                     std::size_t input_size,
                     gdf_dtype input_dtype,
                     std::string input_name);

    std::shared_ptr<gdf_column> build(std::size_t size,
                                      gdf_dtype dtype,
                                      gdf_size_type null_count,
                                      gdf_time_unit time_unit,
                                      std::string&& name);

    bool operator==(const gdf_column& lhs, const gdf_column& rhs);

} // namespace blazingdb
} // namespace test

#endif //BLAZINGDB_COMMUNICATION_TEST_GDF_COLUMN_H
