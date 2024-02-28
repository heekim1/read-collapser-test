#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "consensus/cnn-consensus-strategy.h"
#include "consensus/deep-learning-consensus-caller.h"
#include "read-record.h"

using namespace bfx::read_collapser;

std::shared_ptr<bam1_t> make_fake_read();

class MyConsensusReadAccumulator : public ISink<ConsensusRead> {
 public:
  std::vector<ConsensusRead> Reads;
  void HandleWork(const ConsensusRead& workItem) { Reads.push_back(workItem); }
};

TEST_CASE("Batch") {
  auto model = std::filesystem::path(TEST_RESOURCE_DIR);
  model.append("model.onnx");
  std::basic_string<char> seq_a = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
  std::basic_string<char> seq_b = "GTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGT";
  auto my_sink = std::make_shared<MyConsensusReadAccumulator>();
  {
    auto cnn_consensus = std::make_unique<CnnConsensusStrategy>(model, false, 1);
    AlignmentOptions alignment_opts(10, 8, 8, 6);
    auto consensus_worker =
        std::make_shared<DeepLearningConsensusCaller>(std::move(cnn_consensus), 1, 2, alignment_opts);
    consensus_worker->AddSink(my_sink);

    // feed it a few reads, which it will flush out when it goes out of scope
    std::vector<uint8_t> scores;
    scores.insert(scores.end(), 64, 20);
    // std::vector<uint8_t> scores{20, 20, 20, 20, 20, 20, 20, 20};
    std::vector<bfx::io::CigarEntry> cigar{bfx::io::CigarEntry(bfx::io::CigarOp::ReferenceMatch, 64)};
    consensus_worker->HandleWork(
        {std::make_shared<bfx::io::ReadRecord>(100, 164, seq_a.c_str(), cigar, make_fake_read(), scores, "A"),
         std::make_shared<bfx::io::ReadRecord>(100, 164, seq_a.c_str(), cigar, make_fake_read(), scores, "A"),
         std::make_shared<bfx::io::ReadRecord>(100, 164, seq_a.c_str(), cigar, make_fake_read(), scores, "A"),
         std::make_shared<bfx::io::ReadRecord>(100, 164, seq_a.c_str(), cigar, make_fake_read(), scores, "A")

        });
    REQUIRE(my_sink->Reads.size() == 0);  // batching will not pass the reads through just yet
    consensus_worker->HandleWork(
        {std::make_shared<bfx::io::ReadRecord>(100, 164, seq_b.c_str(), cigar, make_fake_read(), scores, "B"),
         std::make_shared<bfx::io::ReadRecord>(100, 164, seq_b.c_str(), cigar, make_fake_read(), scores, "B"),
         std::make_shared<bfx::io::ReadRecord>(100, 164, seq_b.c_str(), cigar, make_fake_read(), scores, "B"),
         std::make_shared<bfx::io::ReadRecord>(100, 164, seq_b.c_str(), cigar, make_fake_read(), scores, "B")

        });
  }
  REQUIRE(my_sink->Reads.size() == 2);  // now the reads will pass through
  REQUIRE(my_sink->Reads[0].Bases() == seq_a.c_str());
  REQUIRE(my_sink->Reads[0].ReadName() == "A-0-0-4-0-4");

  REQUIRE(my_sink->Reads[1].Bases() == seq_b.c_str());
  REQUIRE(my_sink->Reads[1].ReadName() == "B-0-0-4-0-4");
}

TEST_CASE("Batch with default min depth") {
  auto model = std::filesystem::path(TEST_RESOURCE_DIR);
  model.append("model.onnx");
  auto cnn_consensus = std::make_unique<CnnConsensusStrategy>(model, false, 1);
  AlignmentOptions alignment_opts(10, 8, 8, 6);
  DeepLearningConsensusCaller cnn_caller = DeepLearningConsensusCaller(std::move(cnn_consensus), 1, 2, alignment_opts);
  int expected_min_depth = 2;
  REQUIRE(cnn_caller.GetMinDepth() == expected_min_depth);
}

TEST_CASE("Batch with min depth") {
  auto model = std::filesystem::path(TEST_RESOURCE_DIR);
  model.append("model.onnx");
  auto cnn_consensus = std::make_unique<CnnConsensusStrategy>(model, false, 1);
  AlignmentOptions alignment_opts(10, 8, 8, 6);
  int min_depth = 3;
  DeepLearningConsensusCaller cnn_caller =
      DeepLearningConsensusCaller(std::move(cnn_consensus), 1, 2, alignment_opts, min_depth);
  REQUIRE(cnn_caller.GetMinDepth() == min_depth);
}
