#include <catch2/catch_test_macros.hpp>

#include "majority-voting-consensus-generator.h"

using namespace bfx::read_collapser;

std::shared_ptr<bam1_t> make_fake_read() {
  auto read = new bam1_t();
  read->data = new uint8_t[]{0};
  return std::shared_ptr<bam1_t>(read);
}

TEST_CASE("Min Depth", "[majority_voting]") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 108, "ACGTACGT", {{bfx::io::ReferenceMatch, 8}}, make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 103, "ACG", {{bfx::io::ReferenceMatch, 3}}, make_fake_read(), {20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(105, 108, "CGT", {{bfx::io::ReferenceMatch, 3}}, make_fake_read(), {20, 20, 20})));
  int minDepth = 2;
  MajorityVotingConsensusGenerator consensusGenerator(0.5, minDepth, 1, {10, 8, 8, 6}, nullptr);
  consensusGenerator.setDepth(minDepth);
  auto result = consensusGenerator.DoVoting(reads);
  REQUIRE(get<0>(result) == "ACGCGT");
}

TEST_CASE("Min Depth With wiggle", "[majority_voting]") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      99, 108, "TACGTACGT", {{bfx::io::ReferenceMatch, 9}}, make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 108, "ACGTACGT", {{bfx::io::ReferenceMatch, 8}}, make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 109, "ACGTACGTA", {{bfx::io::ReferenceMatch, 9}}, make_fake_read(), {20, 20, 20, 20, 20, 20, 20, 20, 20})));
  MajorityVotingConsensusGenerator consensusGenerator1(0.5, 2, 1, {10, 8, 8, 6}, nullptr);
  auto result = consensusGenerator1.DoVoting(reads);
  REQUIRE(get<0>(result) == "ACGTACGT");
}

TEST_CASE("Max QScore", "[majority_voting]") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 104, "ACGT", {{bfx::io::ReferenceMatch, 4}}, make_fake_read(), {20, 20, 20, 20})));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 104, "ACGT", {{bfx::io::ReferenceMatch, 4}}, make_fake_read(), {10, 40, 30, 20})));
  MajorityVotingConsensusGenerator consensusGenerator(0.5, 0, 1, {10, 8, 8, 6}, nullptr);
  auto result = consensusGenerator.DoVoting(reads);
  // All Q-scores are Q40
  for (size_t i = 0; i < 4; i++) {
    REQUIRE(get<1>(result)[0] == 40);
  }
}

TEST_CASE("Simple Consensus", "[majority_voting]") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "CCGTTCGTACGTACGG", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "TCGTGCGTACGTACGC", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  float majorityRatio = 0.5;
  MajorityVotingConsensusGenerator consensusGenerator(majorityRatio, 0, 1, {10, 8, 8, 6}, nullptr);
  auto result = consensusGenerator.DoVoting(reads);
  REQUIRE(result.qscores().front() == 0);
  REQUIRE(result.qscores().at(4) == 0);
  REQUIRE(result.qscores().back() == 0);
  // add 2 more copies of the last read
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "TCGTGCGTACGTACGC", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "TCGTGCGTACGTACGC", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  result = consensusGenerator.DoVoting(reads);
  for (const auto& q : result.qscores()) {
    REQUIRE(q == 40);
  }
  majorityRatio = 0.7;
  consensusGenerator.setMajorityRatio(majorityRatio);
  result = consensusGenerator.DoVoting(reads);
  REQUIRE(result.qscores().front() == 0);
  REQUIRE(result.qscores().at(4) == 0);
  REQUIRE(result.qscores().back() == 0);
}

TEST_CASE("Position wiggle handled", "[majority_voting]") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(102, 118, "GTACGTACGTACGTAC", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(101, 117, "CGTACGTACGTACGTA", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  MajorityVotingConsensusGenerator consensusGenerator(0.5, 0, 1, {10, 8, 8, 6}, nullptr);
  auto result = consensusGenerator.DoVoting(reads);
  REQUIRE(get<0>(result) == "CGTACGTACGTACGTA");
}

TEST_CASE("Simple deletion", "[majority_voting]") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTCGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Deletion, 1}, {bfx::io::ReferenceMatch, 11}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTCGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Deletion, 1}, {bfx::io::ReferenceMatch, 11}}, make_fake_read())));
  MajorityVotingConsensusGenerator consensusGenerator(0.5, 0, 1, {10, 8, 8, 6}, nullptr);
  auto result = consensusGenerator.DoVoting(reads);
  // second A is removed
  REQUIRE(get<0>(result) == "ACGTCGTACGTACGT");
  REQUIRE(result.cigar()[0] == bfx::io::CigarEntry{bfx::io::ReferenceMatch, 4});
  REQUIRE(result.cigar()[1] == bfx::io::CigarEntry{bfx::io::Deletion, 1});
  REQUIRE(result.cigar()[2] == bfx::io::CigarEntry{bfx::io::ReferenceMatch, 11});
  REQUIRE(result.cigar().size() == 3);
  // now make it a minorioty
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  result = consensusGenerator.DoVoting(reads);
  REQUIRE(get<0>(result) == "ACGTACGTACGTACGT");
}

TEST_CASE("Simple deletion with super_majority_deletion_treshold", "[majority_voting]") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTCGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Deletion, 1}, {bfx::io::ReferenceMatch, 11}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTCGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Deletion, 1}, {bfx::io::ReferenceMatch, 11}}, make_fake_read())));

  float super_majority_deletion_threshold = 0.5;
  MajorityVotingConsensusGenerator consensusGenerator(0.5, 0, super_majority_deletion_threshold, 1, {10, 8, 8, 6},
                                                      nullptr);
  auto result = consensusGenerator.DoVoting(reads);
  // second A is removed
  REQUIRE(get<0>(result) == "ACGTCGTACGTACGT");
  REQUIRE(result.cigar()[0] == bfx::io::CigarEntry{bfx::io::ReferenceMatch, 4});
  REQUIRE(result.cigar()[1] == bfx::io::CigarEntry{bfx::io::Deletion, 1});
  REQUIRE(result.cigar()[2] == bfx::io::CigarEntry{bfx::io::ReferenceMatch, 11});
  REQUIRE(result.cigar().size() == 3);
  // now make it a minorioty
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  result = consensusGenerator.DoVoting(reads);
  REQUIRE(get<0>(result) == "ACGTACGTACGTACGT");
}

TEST_CASE("Simple insertion", "[majority_voting]") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTAAACGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 2}, {bfx::io::ReferenceMatch, 12}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTAAACGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 2}, {bfx::io::ReferenceMatch, 12}}, make_fake_read())));
  MajorityVotingConsensusGenerator consensusGenerator(0.5, 0, 1, {10, 8, 8, 6}, nullptr);
  auto result = consensusGenerator.DoVoting(reads);
  REQUIRE(get<0>(result) == "ACGTAAACGTACGTACGT");
  REQUIRE(result.cigar()[0] == bfx::io::CigarEntry{bfx::io::ReferenceMatch, 4});
  REQUIRE(result.cigar()[1] == bfx::io::CigarEntry{bfx::io::Insert, 2});
  REQUIRE(result.cigar()[2] == bfx::io::CigarEntry{bfx::io::ReferenceMatch, 12});
  REQUIRE(result.cigar().size() == 3);
  // now make it a minority
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  result = consensusGenerator.DoVoting(reads);
  REQUIRE(get<0>(result) == "ACGTACGTACGTACGT");
}

TEST_CASE("Insertions of different lengths", "[majority_voting]") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTAAACGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 2}, {bfx::io::ReferenceMatch, 12}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTAAACGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 2}, {bfx::io::ReferenceMatch, 12}}, make_fake_read())));
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTAAAACGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 3}, {bfx::io::ReferenceMatch, 12}}, make_fake_read())));
  MajorityVotingConsensusGenerator consensusGenerator(0.5, 0, 1, {10, 8, 8, 6}, nullptr);
  auto result = consensusGenerator.DoVoting(reads);
  REQUIRE(get<0>(result) == "ACGTAAACGTACGTACGT");
}

TEST_CASE("Different insertion sequences", "[majority_voting]") {
  std::vector<bfx::io::ReadRecordPtr> reads;
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(
      new bfx::io::ReadRecord(100, 116, "ACGTACGTACGTACGT", {{bfx::io::ReferenceMatch, 16}}, make_fake_read())));
  // insertion is T
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTTACGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 1}, {bfx::io::ReferenceMatch, 12}}, make_fake_read())));
  // insertion is AA
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTAAACGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 2}, {bfx::io::ReferenceMatch, 12}}, make_fake_read())));
  // insertion is AAT
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTAATACGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 3}, {bfx::io::ReferenceMatch, 12}}, make_fake_read())));
  // insertion is ATAT
  reads.push_back(std::shared_ptr<bfx::io::ReadRecord>(new bfx::io::ReadRecord(
      100, 116, "ACGTATATACGTACGTACGT",
      {{bfx::io::ReferenceMatch, 4}, {bfx::io::Insert, 4}, {bfx::io::ReferenceMatch, 12}}, make_fake_read())));
  MajorityVotingConsensusGenerator consensusGenerator(0.5, 0, 1, {10, 8, 8, 6}, nullptr);
  auto result = consensusGenerator.DoVoting(reads);
  REQUIRE(get<0>(result) == "ACGTAATACGTACGTACGT");
}
