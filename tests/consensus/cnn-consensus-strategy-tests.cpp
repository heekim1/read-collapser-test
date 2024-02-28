#include <catch2/catch_test_macros.hpp>

#include "consensus/cnn-consensus-strategy.h"

using namespace bfx::read_collapser;

// Test fixture struct
struct CnnConsensusStrategyTestFixture {
  arma::Mat<uint8_t> bases;
  arma::Mat<uint8_t> qscores;
  arma::Mat<uint8_t> strands;
  arma::Mat<float> expected_features;
  arma::Cube<uint8_t> cluster;

  CnnConsensusStrategyTestFixture() {
    // Initialization code
    bases = {{1, 2, 0, 4, 4, 3}, 
             {0, 2, 4, 4, 0, 3}, 
             {1, 2, 4, 4, 4, 2}};

    qscores = {{80, 90, 90, 100, 100, 50}, {90, 100, 100, 100, 100, 60}, {100, 70, 60, 80, 100, 100}};

    strands = {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};

    expected_features = {
    #      +DEL      +A        +C         +G      +T      -DEL     -A       -C         -G        -T      TOT 
        {1.0f / 3, 1.0f / 3,        0,        0,        0,    0, 1.0f / 3,        0,        0,        0,   3}, 
        {       0,        0, 2.0f / 3,        0,        0,    0,        0, 1.0f / 3,        0,        0,   3},
        {1.0f / 3,        0,        0,        0, 1.0f / 3,    0,        0,        0,        0, 1.0f / 3,   3}, 
        {       0,        0,        0,        0, 2.0f / 3,    0,        0,        0,        0, 1.0f / 3,   3},
        {1.0f / 3,        0,        0,        0, 1.0f / 3,    0,        0,        0,        0, 1.0f / 3,   3}, 
        {       0,        0,        0, 2.0f / 3,        0,    0,        0, 1.0f / 3,        0,        0,   3}
      };

    cluster = arma::join_slices(bases, qscores);
    cluster = arma::join_slices(cluster, strands);
  }

  ~CnnConsensusStrategyTestFixture() {}
};

TEST_CASE_METHOD(CnnConsensusStrategyTestFixture, "calculateFeatureTest") {
  arma::Mat<float> features = CnnConsensusStrategy::CalculateFeature(cluster, 7);
  arma::Mat<float> expected_result = expected_features.t();
  REQUIRE(arma::approx_equal(features, expected_result, "absdiff", 1e-5));
}

TEST_CASE("CnnConsensusStrategy") {
  arma::Mat<float> mat = arma::ones<arma::Mat<float>>(3, 3);
  arma::umat index = {{10, 0, 0}, {0, 5, 0}, {0, 0, 1}};
  arma::Mat<float> actual_result = CnnConsensusStrategy::MatrixWhere(mat, index, -1);
  arma::Mat<float> expected_result = {{-1, 1, 1}, {1, -1, 1}, {1, 1, -1}};

  REQUIRE(arma::approx_equal(actual_result, expected_result, "absdiff", 1e-5));
}

TEST_CASE_METHOD(CnnConsensusStrategyTestFixture, "createBatchesTest") {
  arma::Col<float> expected_features_vector = arma::vectorise(expected_features);
  std::vector<float> expected_batch_data(expected_features_vector.begin(), expected_features_vector.end());

  std::vector<arma::Cube<uint8_t>> clusters = {cluster, cluster};
  std::vector<float> batch_features;
  CnnConsensusStrategy::CreateBatches(clusters, batch_features, NUMFEATUREWITHOUTQSCORE, 7);

  std::vector<float> first_cluster_feature(batch_features.begin(), batch_features.begin() + batch_features.size() / 2);
  REQUIRE(first_cluster_feature == expected_batch_data);

  std::vector<float> second_cluster_feature(batch_features.begin() + batch_features.size() / 2, batch_features.end());
  REQUIRE(second_cluster_feature == expected_batch_data);
}

// Test fixture struct
struct CnnConsensusStrategyPartialReadTestFixture {
  arma::Mat<uint8_t> bases;
  arma::Mat<uint8_t> qscores;
  arma::Mat<uint8_t> strands;
  arma::Mat<float> expected_features;
  arma::Cube<uint8_t> cluster;

  CnnConsensusStrategyPartialReadTestFixture() {
    // Initialization code
    bases = {{1, 2, 0, 4, 4, 3}, {7, 2, 4, 4, 0, 3}, {1, 2, 4, 4, 4, 7}};

    qscores = {{80, 90, 90, 100, 100, 50}, {0, 100, 100, 100, 100, 60}, {100, 70, 60, 80, 100, 0}};

    strands = {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};

    expected_features = {
        {0, 1.0f / 2, 0, 0, 0, 0, 1.0f / 2, 0, 0, 0, 2},         {0., 0., 2.0f / 3, 0, 0, 0, 0, 1.0f / 3, 0, 0, 3},
        {1.0f / 3, 0., 0, 0, 1.0f / 3, 0, 0, 0, 0, 1.0f / 3, 3}, {0., 0., 0, 0, 2.0f / 3, 0, 0, 0, 0, 1.0f / 3, 3},
        {1.0f / 3, 0., 0, 0, 1.0f / 3, 0, 0, 0, 0, 1.0f / 3, 3}, {0., 0, 0., 1.0, 0, 0, 0, 0, 0, 0, 2}};

    cluster = arma::join_slices(bases, qscores);
    cluster = arma::join_slices(cluster, strands);
  }

  ~CnnConsensusStrategyPartialReadTestFixture() {}
};

TEST_CASE_METHOD(CnnConsensusStrategyPartialReadTestFixture, "calculatePartialReadFeatureTest") {
  arma::Mat<float> features = CnnConsensusStrategy::CalculateFeature(cluster, 7);
  arma::Mat<float> expected_result = expected_features.t();
  REQUIRE(arma::approx_equal(features, expected_result, "absdiff", 1e-5));
}

TEST_CASE_METHOD(CnnConsensusStrategyPartialReadTestFixture, "createPartialReadBatchesTest") {
  arma::Col<float> expected_features_vector = arma::vectorise(expected_features);
  std::vector<float> expected_batch_data(expected_features_vector.begin(), expected_features_vector.end());

  std::vector<arma::Cube<uint8_t>> clusters = {cluster, cluster};
  std::vector<float> batchFeatures;
  CnnConsensusStrategy::CreateBatches(clusters, batchFeatures, NUMFEATUREWITHOUTQSCORE, 7);

  std::vector<float> first_cluster_feature(batchFeatures.begin(), batchFeatures.begin() + batchFeatures.size() / 2);
  REQUIRE(first_cluster_feature == expected_batch_data);

  std::vector<float> second_cluster_feature(batchFeatures.begin() + batchFeatures.size() / 2, batchFeatures.end());
  REQUIRE(second_cluster_feature == expected_batch_data);
}

TEST_CASE("BaseQualitiesToPhredScores") {
  arma::Row<float> baseQualities = {0.991, 0.995, 0.9991, 0.991, 0.9991, 0.99991};
  // Convert base qualities to Phred vector
  auto phredScores = CnnConsensusStrategy::BaseQualitiesToPhredScores(baseQualities);
  auto expectedPhredScores = std::vector<uint8_t>{20, 23, 30, 20, 30, 40};  // "58?5?I";

  REQUIRE(phredScores == expectedPhredScores);
}

TEST_CASE("NumericToDnaBases") {
  arma::Row<arma::uword> calls = {0, 1, 2, 0, 3, 4};
  // Convert numeric bases to Dna bases string
  std::string dnaBases = CnnConsensusStrategy::NumericToDnaBases(calls);
  std::string expectedDnaBases = "-AC-GT";
  REQUIRE(dnaBases == expectedDnaBases);
}

TEST_CASE("RemoveGapsWithQuality") {
  arma::Row<float> baseQualities = {0.991, 0.995, 0.9991, 0.991, 0.9991, 0.99991};
  // Convert base qualities to Phred+33 characters string
  arma::Row<arma::uword> calls = {0, 1, 2, 0, 3, 4};
  // Convert numeric bases to Dna bases string
  std::string dnaBases = CnnConsensusStrategy::NumericToDnaBases(calls);

  auto scores = CnnConsensusStrategy::BaseQualitiesToPhredScores(baseQualities);

  size_t consensus_length = calls.size();
  CnnConsensusStrategy::RemoveGapsWithQuality(dnaBases, scores, consensus_length);

  std::string expectedDnaBases = "ACGT";
  auto expectedPhredScores = std::vector<uint8_t>{23, 30, 30, 40};  // 8??I";

  REQUIRE(dnaBases == expectedDnaBases);
  REQUIRE(scores == expectedPhredScores);
}

TEST_CASE("NormalizeBaseProb") {
  // Initialize softmaxValue include base call probabilities
  arma::Cube<float> softmaxValue(5, 5, 1);

  // Fill the first slice (i=0) with example data
  softmaxValue.slice(0) = {{0.2, 0.0, 0.0, 0.0, 0.8},
                           {0.8, 0.0, 0.0, 0.0, 0.2},
                           {0.0, 0.0, 0.0, 0.0, 1},
                           {0.6, 0.0, 0.0, 0.0, 0.4},
                           {0.1, 0.0, 0.0, 0.0, 0.9}};

  // Initialize arma::frowvec num_pass_per_column
  arma::frowvec num_pass_per_column = {10, 5, 10, 10, 10};

  // Initialize n_slice and kScaleDownGapScore
  size_t n_slice = 0;

  CnnConsensusStrategy::NormalizeBaseProb(softmaxValue, num_pass_per_column, n_slice);

  // Initialize the expected softmaxValue
  arma::Cube<float> expected_softmaxValue(5, 5, 1);
  expected_softmaxValue.slice(0) = {{0.0476, 0.0, 0.0, 0.0, 0.9524},
                                    {0.8, 0.0, 0.0, 0.0, 0.2},
                                    {0.0, 0.0, 0.0, 0.0, 1},
                                    {0.6, 0.0, 0.0, 0.0, 0.4},
                                    {0.0217, 0.0, 0.0, 0.0, 0.9783}};
  REQUIRE(arma::approx_equal(softmaxValue, expected_softmaxValue, "absdiff", 1e-4));
}

TEST_CASE("UpdateBasedProbWhereGapIsMajority") {
  // Initialize arma::Mat<float> base_counts
  arma::Mat<float> base_pct = {{0.0, 0.0, 0.0, 0.0, 1},
                               {0.8, 0.0, 0.0, 0.0, 0.2},
                               {0.0, 0.0, 0.0, 0.0, 1},
                               {0.6, 0.0, 0.0, 0.0, 0.4},
                               {0.0, 0.0, 0.0, 0.0, 1}};

  // Initialize the cube
  arma::Cube<float> softmaxValue(5, 5, 1);

  // Fill the first slice (i=0) with example data
  softmaxValue.slice(0) = {{0.0, 0.0, 0.0, 0.0, 1},
                           {0.8, 0.0, 0.0, 0.0, 0.2},
                           {0.0, 0.0, 0.0, 0.0, 1},
                           {0.6, 0.0, 0.0, 0.0, 0.4},
                           {0.0, 0.0, 0.0, 0.0, 1}};

  // Initialize arma::frowvec num_pass_per_column
  arma::frowvec num_pass_per_column = {10, 5, 10, 10, 10};

  // Initialize size_t n_slice
  size_t n_slice = 0;

  // Call UpdateBasedProbAndCallWhereGapIsMajority
  CnnConsensusStrategy::UpdateBasedProbWhereGapIsMajority(base_pct, softmaxValue, num_pass_per_column, n_slice);

  // Initialize the expected softmaxValue
  arma::Cube<float> expected_softmaxValue(5, 5, 1);
  expected_softmaxValue.slice(0) = {{0.0, 0.0, 0.0, 0.0, 1},
                                    {1.0, 0.0, 0.0, 0.0, 0.0},
                                    {0.0, 0.0, 0.0, 0.0, 1},
                                    {0.6, 0.0, 0.0, 0.0, 0.4},
                                    {0.0, 0.0, 0.0, 0.0, 1}};
  REQUIRE(arma::approx_equal(softmaxValue, expected_softmaxValue, "absdiff", 1e-5));
}

TEST_CASE("UpdateBasedProbWhereBasePctMeetsMinAF") {
  // Initialize arma::Mat<float> base_counts
  arma::Mat<float> base_pct = {{0.0, 0.0, 0.0, 0.0, 1},
                               {0.8, 0.0, 0.0, 0.0, 0.2},
                               {0.0, 0.0, 0.0, 0.0, 1},
                               {0.6, 0.0, 0.0, 0.0, 0.4},
                               {0.0, 0.0, 0.0, 0.0, 1}};

  // Initialize the cube
  arma::Cube<float> softmaxValue(5, 5, 1);

  // Fill the first slice (i=0) with example data
  softmaxValue.slice(0) = {{0.1, 0.0, 0.0, 0.0, 1},
                           {0.8, 0.0, 0.0, 0.0, 0.2},
                           {0.0, 0.0, 0.0, 0.2, 1},
                           {0.6, 0.0, 0.0, 0.0, 0.4},
                           {0.1, 0.0, 0.0, 0.0, 1}};

  // Initialize arma::frowvec num_pass_per_column
  arma::frowvec num_pass_per_column = {10, 5, 10, 10, 10};

  // Initialize size_t n_slice
  size_t n_slice = 0;

  // Call UpdateBasedProbAndCallWhereBasePctIsGreaterThanOrEqualToMinAF
  CnnConsensusStrategy::UpdateBasedProbWhereBasePctMeetsMinAF(base_pct, softmaxValue, num_pass_per_column, n_slice);

  // Initialize the expected softmaxValue
  arma::Cube<float> expected_softmaxValue(5, 5, 1);
  expected_softmaxValue.slice(0) = {{0.0, 0.0, 0.0, 0.0, 1},
                                    {0.8, 0.0, 0.0, 0.0, 0.2},
                                    {0.0, 0.0, 0.0, 0.0, 1},
                                    {0.6, 0.0, 0.0, 0.0, 0.4},
                                    {0.0, 0.0, 0.0, 0.0, 1}};
  REQUIRE(arma::approx_equal(softmaxValue, expected_softmaxValue, "absdiff", 1e-5));
}

TEST_CASE("UpdateBaseProbWhereGapIsReplaced") {
  // Initialize arma::Mat<float> base_counts
  arma::Mat<float> base_pct = {{0.0, 0.5, 0.0, 0.0, 0.5},
                               {0.0, 0.0, 0.0, 0.5, 0.5},
                               {0.0, 0.5, 0.0, 0.5, 0.0},
                               {0.0, 0.0, 0.5, 0.5, 0.0},
                               {0.5, 0.0, 0.0, 0.0, 0.5}};

  // Initialize arma::frowvec num_pass_per_column
  arma::frowvec num_pass_per_column = {2, 2, 2, 2, 2};

  // Initialize the cube
  arma::Cube<float> softmaxValue(5, 5, 1);

  // Fill the first slice (i=0) with example data
  softmaxValue.slice(0) = {{0.0, 0.5, 0.0, 0.0, 0.5},
                           {0.0, 0.0, 0.0, 0.5, 0.5},
                           {0.0, 0.5, 0.0, 0.5, 0.0},
                           {0.0, 0.0, 0.5, 0.5, 0.0},
                           {0.5, 0.0, 0.0, 0.0, 0.5}};
  size_t n_slice = 0;

  CnnConsensusStrategy::UpdateBaseProbWhereGapIsReplaced(base_pct, softmaxValue, num_pass_per_column, n_slice);

  // Initialize the expected softmaxValue
  arma::Cube<float> expected_softmaxValue(5, 5, 1);
  expected_softmaxValue.slice(0) = {{0.0, 0.5, 0.0, 0.0, 0.5},
                                    {0.0, 0.0, 0.0, 0.5, 0.5},
                                    {0.0, 0.5, 0.0, 0.5, 0.0},
                                    {0.0, 0.0, 0.5, 0.5, 0.0},
                                    {0.0, 0.0, 0.0, 0.0, 0.2057}};
  REQUIRE(arma::approx_equal(softmaxValue, expected_softmaxValue, "absdiff", 1e-4));
}

TEST_CASE("UpdateBaseProbWhereMajorityBaseCountIsTwo") {
  // Initialize arma::Mat<float> base_counts
  arma::Mat<float> base_pct = {{0.0, 0.5, 0.0, 0.0, 0.5},
                               {0.0, 0.0, 0.0, 0.5, 0.5},
                               {0.0, 0.5, 0.0, 0.5, 0.0},
                               {0.0, 0.0, 0.5, 0.5, 0.0},
                               {0.5, 0.0, 0.0, 0.0, 0.5}};

  // Initialize the cube
  arma::Cube<float> softmaxValue(5, 5, 1);

  // Fill the first slice (i=0) with example data
  softmaxValue.slice(0) = {{0.0, 0.5, 0.0, 0.0, 0.5},
                           {0.0, 0.0, 0.0, 0.5, 0.5},
                           {0.0, 0.5, 0.0, 0.5, 0.0},
                           {0.0, 0.0, 0.5, 0.5, 0.0},
                           {0.5, 0.0, 0.0, 0.0, 0.5}};

  size_t n_slice = 0;

  // Initialize arma::frowvec num_pass_per_column
  arma::frowvec num_pass_per_column = {2, 2, 2, 2, 2};

  // Call UpdateBaseProbWhereMajorityBaseCountIsTwo
  CnnConsensusStrategy::UpdateBaseProbWhereMajorityBaseCountIsTwo(base_pct, softmaxValue, num_pass_per_column, n_slice);

  // Initialize the expected softmaxValue
  arma::Cube<float> expected_softmaxValue(5, 5, 1);
  expected_softmaxValue.slice(0) = {{0.0, 0.20567, 0.0, 0.0, 0.0},
                                    {0.0, 0.0, 0.0, 0.20567, 0.0},
                                    {0.0, 0.20567, 0.0, 0.0, 0.0},
                                    {0.0, 0.0, 0.20567, 0.0, 0.0},
                                    {0.0, 0.0, 0.0, 0.0, 0.20567}};
  REQUIRE(arma::approx_equal(softmaxValue, expected_softmaxValue, "absdiff", 1e-5));
}

TEST_CASE("UpdateBaseProbWhereMajorityBaseCountIsOne") {
  // Initialize arma::Mat<float> base_counts
  arma::Mat<float> base_pct = {{0.0, 0.5, 0.0, 0.0, 0.5},
                               {0.0, 0.0, 0.0, 0.5, 0.5},
                               {0.0, 1.0, 0.0, 0.0, 0.0},
                               {0.0, 0.0, 1.0, 0.0, 0.0},
                               {0.5, 0.0, 0.0, 0.0, 0.5}};

  // Initialize the cube
  arma::Cube<float> softmaxValue(5, 5, 1);

  // Fill the first slice (i=0) with example data
  softmaxValue.slice(0) = {{0.0, 0.5, 0.0, 0.0, 0.5},
                           {0.0, 0.0, 0.0, 0.5, 0.5},
                           {0.0, 1.0, 0.0, 0.0, 0.0},
                           {0.0, 0.0, 1.0, 0.0, 0.0},
                           {0.5, 0.0, 0.0, 0.0, 0.5}};

  size_t n_slice = 0;

  // Initialize arma::frowvec num_pass_per_column
  arma::frowvec num_pass_per_column = {2, 2, 1, 1, 2};

  // Call UpdateBaseProbWhereMajorityBaseCountIsTwo
  CnnConsensusStrategy::UpdateBaseProbWhereMajorityBaseCountIsOne(base_pct, softmaxValue, num_pass_per_column, n_slice);

  // Initialize the expected softmaxValue
  arma::Cube<float> expected_softmaxValue(5, 5, 1);
  expected_softmaxValue.slice(0) = {{0.0, 0.5, 0.0, 0.0, 0.5},
                                    {0.0, 0.0, 0.0, 0.5, 0.5},
                                    {0.0, 0.20567, 0.0, 0.0, 0.0},
                                    {0.0, 0.0, 0.20567, 0.0, 0.0},
                                    {0.5, 0.0, 0.0, 0.0, 0.5}};
  REQUIRE(arma::approx_equal(softmaxValue, expected_softmaxValue, "absdiff", 1e-5));
}
