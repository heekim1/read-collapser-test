#include <catch2/catch_test_macros.hpp>

#include "armadillo"
#include "msa_bam/bam-to-msa-converter.h"

using namespace bfx::read_collapser;

std::shared_ptr<bam1_t> make_fake_read();

TEST_CASE("bfx::io::ReferenceMatch") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 108, "TACGTACGT", {{bfx::io::ReferenceMatch, 9}}, make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 108, "ACGTACGT", {{bfx::io::ReferenceMatch, 8}}, make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 109, "ACGTACGTA", {{bfx::io::ReferenceMatch, 9}}, make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20})));

  BAMtoMSAConverter converter;
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);

  std::vector<std::vector<uint8_t>> expected_msa = {
      {1, 2, 3, 4, 1, 2, 3, 4}, {1, 2, 3, 4, 1, 2, 3, 4}, {1, 2, 3, 4, 1, 2, 3, 4}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("DeletionAndInsert") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(99, 108, "ATGAGCTA",
                                                                               {{bfx::io::ReferenceMatch, 3},
                                                                                {bfx::io::Deletion, 1},
                                                                                {bfx::io::ReferenceMatch, 2},
                                                                                {bfx::io::Insert, 1},
                                                                                {bfx::io::ReferenceMatch, 2}},
                                                                               make_fake_read(),
                                                                               {20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 107, "ATGAGTA", {{bfx::io::ReferenceMatch, 3}, {bfx::io::Deletion, 1}, {bfx::io::ReferenceMatch, 4}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20})));

  BAMtoMSAConverter converter;
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);

  std::vector<std::vector<uint8_t>> expected_msa = {{1, 4, 3, 1, 3, 2, 4, 1}, {1, 4, 3, 1, 3, 0, 4, 1}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("Delete bases (greater than 1) And bfx::io::Insert") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(99, 108, "ATGAGCTA",
                                                                               {{bfx::io::ReferenceMatch, 3},
                                                                                {bfx::io::Deletion, 2},
                                                                                {bfx::io::ReferenceMatch, 2},
                                                                                {bfx::io::Insert, 1},
                                                                                {bfx::io::ReferenceMatch, 2}},
                                                                               make_fake_read(),
                                                                               {20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 107, "ATGAGTA", {{bfx::io::ReferenceMatch, 3}, {bfx::io::Deletion, 1}, {bfx::io::ReferenceMatch, 4}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20})));

  BAMtoMSAConverter converter;
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);

  std::vector<std::vector<uint8_t>> expected_msa = {{1, 4, 3, 0, 1, 3, 2, 4}, {1, 4, 3, 1, 3, 4, 0, 1}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("MultipleInserts") {
  std::vector<bfx::io::ReadRecordPtr> reads;

  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 110, "ACGTAAAAAC", {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 1}, {bfx::io::ReferenceMatch, 5}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20, 20})));

  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 112, "ACGTATAAAACT", {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 2}, {bfx::io::ReferenceMatch, 6}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20})));

  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 110, "ACGTAAAAGC", {{bfx::io::ReferenceMatch, 8}, {bfx::io::Insert, 1}, {bfx::io::ReferenceMatch, 1}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20, 20})));

  BAMtoMSAConverter converter;
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);

  std::vector<std::vector<uint8_t>> expected_msa = {
      {1, 2, 3, 4, 1, 0, 1, 1, 1, 1, 0, 2}, {1, 2, 3, 4, 1, 4, 1, 1, 1, 1, 0, 2}, {1, 2, 3, 4, 0, 0, 1, 1, 1, 1, 3, 2}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("3-prime bfx::io::SoftClip When remove-soft-clip is true") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(99, 108, "TACGTACGT", {{bfx::io::ReferenceMatch, 6}, {bfx::io::SoftClip, 3}},
                              make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 108, "ACGTACGT", {{bfx::io::ReferenceMatch, 5}, {bfx::io::SoftClip, 3}},
                              make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 109, "ACGTACGTA", {{bfx::io::ReferenceMatch, 5}, {bfx::io::SoftClip, 4}},
                              make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20})));

  BAMtoMSAConverter converter;  // default remove_soft_clips = true
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);

  std::vector<std::vector<uint8_t>> expected_msa = {{1, 2, 3, 4, 1}, {1, 2, 3, 4, 1}, {1, 2, 3, 4, 1}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("5-prime bfx::io::SoftClip When remove-soft-clip is true") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(99, 108, "TACGTACGT", {{bfx::io::SoftClip, 2}, {bfx::io::ReferenceMatch, 7}},
                              make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 108, "ACGTACGT", {{bfx::io::SoftClip, 1}, {bfx::io::ReferenceMatch, 7}},
                              make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 109, "ACGTACGTA", {{bfx::io::SoftClip, 1}, {bfx::io::ReferenceMatch, 8}},
                              make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20})));

  BAMtoMSAConverter converter;  // default remove_soft_clips = true
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);

  std::vector<std::vector<uint8_t>> expected_msa = {
      {3, 4, 1, 2, 3, 4, 7}, {2, 3, 4, 1, 2, 3, 4}, {2, 3, 4, 1, 2, 3, 4}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("GetNonEmptyColumns") {
  std::vector<std::vector<uint8_t>> final_MSA = {{0, 2, 3, 4, 1, 0, 1, 1, 1, 1, 0, 2, 7},
                                                 {0, 2, 3, 4, 1, 4, 1, 1, 1, 1, 0, 2, 7},
                                                 {0, 2, 3, 4, 0, 0, 1, 1, 1, 1, 3, 2, 7}};

  BAMtoMSAConverter converter;
  std::vector<int> non_empty_columns = converter.GetNonEmptyColumns(final_MSA);
  std::vector<int> expected_non_empty_columns = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  REQUIRE(non_empty_columns == expected_non_empty_columns);
}

TEST_CASE("GetMoreThanOnePassColumns") {
  std::vector<std::vector<uint8_t>> final_MSA = {{0, 2, 3, 4, 1, 0, 1, 1, 1, 1, 0, 2, 7},
                                                 {0, 2, 3, 4, 1, 4, 1, 1, 1, 1, 0, 2, 7},
                                                 {0, 2, 3, 4, 0, 0, 1, 1, 1, 1, 3, 2, 7}};

  BAMtoMSAConverter converter;
  int start = 1;
  int end = 11;
  std::vector<int> more_than_one_pass_columns = converter.GetMoreThanOnePassColumns(final_MSA, start, end);
  std::vector<int> expected_more_than_one_pass_columns = {1, 2, 3, 4, 6, 7, 8, 9, 11};
  REQUIRE(more_than_one_pass_columns == expected_more_than_one_pass_columns);
}

TEST_CASE("GetNonGapColumns when full_read_size >= 1") {
  std::vector<std::vector<uint8_t>> final_MSA = {{0, 2, 3, 4, 1, 0, 1, 0, 2, 7},
                                                 {0, 2, 3, 4, 1, 4, 1, 0, 2, 7},
                                                 {0, 2, 3, 4, 0, 0, 1, 3, 2, 7},
                                                 {0, 2, 3, 4, 1, 7, 7, 7, 7, 7}};

  BAMtoMSAConverter converter;
  int full_read_size = 3;
  int start = 1;
  int end = 9;
  std::vector<int> non_empty_columns = converter.GetNonGapColumns(final_MSA, full_read_size, start, end);
  std::vector<int> expected_non_empty_columns = {1, 2, 3, 4, 6, 8};
  REQUIRE(non_empty_columns == expected_non_empty_columns);
}

TEST_CASE("GetNonGapColumns when full_read_size == 0") {
  std::vector<std::vector<uint8_t>> final_MSA = {{0, 2, 3, 4, 1, 0, 1, 0, 2, 7},
                                                 {0, 2, 3, 4, 1, 4, 1, 0, 2, 7},
                                                 {0, 2, 3, 4, 0, 0, 1, 3, 2, 7},
                                                 {0, 2, 3, 4, 1, 7, 7, 7, 7, 7}};

  BAMtoMSAConverter converter;
  int full_read_size = 0;
  int start = 1;
  int end = 8;
  std::vector<int> non_empty_columns = converter.GetNonGapColumns(final_MSA, full_read_size, start, end);
  std::vector<int> expected_non_empty_columns = {1, 2, 3, 4, 6, 8};
  REQUIRE(non_empty_columns == expected_non_empty_columns);
}

TEST_CASE("TrimAlignmentInfo") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(99, 108, "TACGTACGT", {{bfx::io::SoftClip, 2}, {bfx::io::ReferenceMatch, 7}},
                              make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 108, "ACGTACGT", {{bfx::io::SoftClip, 1}, {bfx::io::ReferenceMatch, 7}},
                              make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 109, "ACGTACGTA", {{bfx::io::SoftClip, 1}, {bfx::io::ReferenceMatch, 8}},
                              make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20})));

  BAMtoMSAConverter converter;  // default remove_soft_clips = true
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);
  alignment_info = converter.TrimAlignmentInfo(alignment_info);

  std::vector<std::vector<uint8_t>> expected_msa = {
      {3, 4, 1, 2, 3, 4, 7}, {2, 3, 4, 1, 2, 3, 4}, {2, 3, 4, 1, 2, 3, 4}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("CheckINSinPartials") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 106, "TACGTACGTACG",
      {{bfx::io::SoftClip, 2}, {bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 2}, {bfx::io::ReferenceMatch, 4}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 107, "AGTATAGCGT",
      {{bfx::io::SoftClip, 1}, {bfx::io::ReferenceMatch, 5}, {bfx::io::Insert, 1}, {bfx::io::ReferenceMatch, 3}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 102, "AGTA", {{bfx::io::SoftClip, 1}, {bfx::io::ReferenceMatch, 3}}, make_fake_read(), {20, 20, 20, 20})));

  BAMtoMSAConverter converter;  // default remove_soft_clips = true
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);

  std::vector<std::vector<uint8_t>> expected_msa = {
      {3, 4, 1, 2, 3, 4, 1, 0, 2, 3}, {3, 4, 1, 0, 0, 4, 1, 3, 2, 3}, {3, 4, 1, 7, 7, 7, 7, 7, 7, 7}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("DeleteGapMajorColumns") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 111, "ATGAGTA", {{bfx::io::ReferenceMatch, 3}, {bfx::io::Deletion, 5}, {bfx::io::ReferenceMatch, 4}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 111, "ATGAGTA", {{bfx::io::ReferenceMatch, 3}, {bfx::io::Deletion, 5}, {bfx::io::ReferenceMatch, 4}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 111, "ATGAGTA", {{bfx::io::ReferenceMatch, 3}, {bfx::io::Deletion, 5}, {bfx::io::ReferenceMatch, 4}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20})));

  BAMtoMSAConverter converter;
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);
  converter.DeleteGapMajorColumns(alignment_info);

  std::vector<std::vector<uint8_t>> expected_msa = {
      {1, 4, 3, 1, 3, 4, 1}, {1, 4, 3, 1, 3, 4, 1}, {1, 4, 3, 1, 3, 4, 1}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("DeleteGapMajorColumnsWithPartials") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(99, 105, "ATGAGTACAG",
                              {{bfx::io::ReferenceMatch, 3},
                               {bfx::io::Insert, 2},
                               {bfx::io::ReferenceMatch, 2},
                               {bfx::io::Insert, 1},
                               {bfx::io::ReferenceMatch, 2}},
                              make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 105, "ATGTAAG", {{bfx::io::ReferenceMatch, 7}}, make_fake_read(), {20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 105, "ATGTAAG", {{bfx::io::ReferenceMatch, 7}}, make_fake_read(), {20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 105, "ATGTAAG", {{bfx::io::ReferenceMatch, 7}}, make_fake_read(), {20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 103, "ATGTA", {{bfx::io::ReferenceMatch, 5}}, make_fake_read(), {20, 20, 20, 20, 20})));

  BAMtoMSAConverter converter;
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);
  converter.DeleteGapMajorColumns(alignment_info);

  std::vector<std::vector<uint8_t>> expected_msa = {{1, 4, 3, 4, 1, 2, 1, 3},
                                                    {1, 4, 3, 4, 1, 0, 1, 3},
                                                    {1, 4, 3, 4, 1, 0, 1, 3},
                                                    {1, 4, 3, 4, 1, 0, 1, 3},
                                                    {1, 4, 3, 4, 1, 7, 7, 7}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("CheckDiscardEmptyReads") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 106, "TACGTACGTACG",
      {{bfx::io::SoftClip, 2}, {bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 2}, {bfx::io::ReferenceMatch, 4}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 107, "AGTATAGCGT",
      {{bfx::io::SoftClip, 1}, {bfx::io::ReferenceMatch, 5}, {bfx::io::Insert, 1}, {bfx::io::ReferenceMatch, 3}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(97, 98, "AG", {{bfx::io::ReferenceMatch, 2}}, make_fake_read(), {20, 20})));

  BAMtoMSAConverter converter;  // default remove_soft_clips = true
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);
  converter.RemoveEmptyReads(alignment_info);

  std::vector<std::vector<uint8_t>> expected_msa = {{3, 4, 1, 2, 3, 4, 1, 0, 2, 3}, {3, 4, 1, 0, 0, 4, 1, 3, 2, 3}};
  REQUIRE(alignment_info.msa == expected_msa);
}

TEST_CASE("RemoveRowsHelper") {
  std::vector<std::vector<uint8_t>> msa = {
      {1, 2, 3, 4, 1, 1, 7}, {1, 2, 3, 4, 2, 2, 7}, {1, 2, 3, 4, 3, 3, 7}, {1, 2, 3, 4, 4, 4, 7}};

  std::vector<size_t> rows = {1, 3};

  BAMtoMSAConverter converter;
  converter.RemoveRowsHelper(msa, rows);

  std::vector<std::vector<uint8_t>> expected_msa = {{1, 2, 3, 4, 1, 1, 7}, {1, 2, 3, 4, 3, 3, 7}};
  REQUIRE(msa == expected_msa);
}

TEST_CASE("RemoveColumnsHelper") {
  std::vector<std::vector<uint8_t>> msa = {
      {1, 2, 3, 4, 1, 1, 7}, {1, 2, 3, 4, 2, 2, 7}, {1, 2, 3, 4, 3, 3, 7}, {1, 2, 3, 4, 4, 4, 7}};

  std::vector<size_t> columns = {1, 3, 5};

  BAMtoMSAConverter converter;
  std::vector<std::vector<uint8_t>> expected_msa = {{1, 3, 1, 7}, {1, 3, 2, 7}, {1, 3, 3, 7}, {1, 3, 4, 7}};
  converter.RemoveColumnsHelper(msa, columns);

  REQUIRE(msa == expected_msa);
}

TEST_CASE("GetGapMajorColumns") {
  std::vector<std::vector<uint8_t>> msa = {
      {1, 0, 3, 4, 1, 1, 0}, {1, 2, 3, 4, 2, 2, 7}, {1, 2, 3, 4, 3, 3, 7}, {1, 2, 3, 4, 4, 4, 7}};

  BAMtoMSAConverter converter;
  std::vector<size_t> expected_columns = {6};
  std::vector<size_t> columns = converter.GetGapMajorColumns(msa);

  REQUIRE(columns == expected_columns);
}

TEST_CASE("SetEffectiveNumPass") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 106, "TACGTACGTACG",
      {{bfx::io::SoftClip, 2}, {bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 2}, {bfx::io::ReferenceMatch, 4}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 107, "AGTATAGCGT",
      {{bfx::io::SoftClip, 1}, {bfx::io::ReferenceMatch, 5}, {bfx::io::Insert, 1}, {bfx::io::ReferenceMatch, 3}},
      make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(97, 98, "AG", {{bfx::io::ReferenceMatch, 2}}, make_fake_read(), {20, 20})));

  BAMtoMSAConverter converter;  // default remove_soft_clips = true
  AlignmentInfo alignment_info = converter.ConvertBAMtoAlignmentInfo(reads);
  converter.SetEffectiveNumPass(alignment_info);

  int expected_effective_num_pass = 2;
  REQUIRE(alignment_info.effective_num_pass == expected_effective_num_pass);
}
