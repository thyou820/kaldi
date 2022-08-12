// gmmbin/gmm-decode-simple.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/simple-decoder.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "fstext/lattice-utils.h"
#include "lat/kaldi-lattice.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::Fst;
    using fst::StdArc;
    using fst::ReadFstKaldiGeneric;

    const char *usage =
        "Decode features using GMM-based model.\n"
        "Viterbi decoding, Only produces linear sequence; any lattice\n"
        "produced is linear\n"
        "\n"
        "Usage:   gmm-decode-simple [options] <model-in> <fst-in> "
        "<features-rspecifier> <words-wspecifier> [<alignments-wspecifier>] "
        "[<lattice-wspecifier>]";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = true; 
    //對聲學模型代價做縮放 (通常聲學模型的代價都會比較大)
    BaseFloat acoustic_scale = 0.1;

    std::string word_syms_filename;
    //beam越大,最終topN的這個N就會越大
    BaseFloat beam = 16.0;
    po.Register("beam", &beam, "Decoding log-likelihood beam");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "Produce output even when final state was not reached");
    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    //聲學模型
    std::string model_in_filename = po.GetArg(1),
        //HCLG.FST
        fst_in_filename = po.GetArg(2),
        //特徵
        feature_rspecifier = po.GetArg(3),
        //輸出的單詞序列
        words_wspecifier = po.GetArg(4),
        //幀對齊的結果 (transistion-id -> phone)
        alignment_wspecifier = po.GetOptArg(5),
        //詞格,topN中的路徑重新生成一個圖
        lattice_wspecifier = po.GetOptArg(6);

    //讀取transition-model(讀取聲學模型)
    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    //讀取fst (HCLG.fst)
    Fst<StdArc> *decode_fst = ReadFstKaldiGeneric(fst_in_filename);

    //輸出單詞序列
    Int32VectorWriter words_writer(words_wspecifier);

    //輸出的對齊序列
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    //Lattice
    CompactLatticeWriter clat_writer(lattice_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    // 讀取特徵
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    //總體似然
    BaseFloat tot_like = 0.0;
    //幀數
    kaldi::int64 frame_count = 0;
    //多少句子解碼成功,多少失敗
    int num_success = 0, num_fail = 0;
    // **核心 (聲明變量) (HCLG.fst, beam)
    SimpleDecoder decoder(*decode_fst, beam);

    //**遍歷音頻
    for (; !feature_reader.Done(); feature_reader.Next()) {

      // utt:音頻id
      std::string utt = feature_reader.Key(); 
      //查看音頻id 可以cout出來看，取消下面程式碼的註解，再,make一次即可
      //std::cout<<"音頻id - :"<<utt<<std::endl;

      // Matrix:音頻特幀的格式、features:音頻特徵內容
      //每列是一幀，一幀有39維的數值(***待定)
      Matrix<BaseFloat> features (feature_reader.Value());
      //std::cout<<"音頻內容 - :"<<features<<std::endl;

      feature_reader.FreeCurrent();
      
      //判斷音頻特徵是否為0，如果音頻特徵是0，num_fail+1
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }

      //聲學模型 (看到gmm通常都是聲學模型)
      DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                             acoustic_scale);

      //解碼 (有了HCLG,聲學模型)
      decoder.Decode(&gmm_decodable);

      //聲明lattice
      VectorFst<LatticeArc> decoded;  // linear FST.

      if ( (allow_partial || decoder.ReachedFinal())
           && decoder.GetBestPath(&decoded) ) {
        //判斷解碼有沒有到達終點 -> 沒有也是會successs -> 但會warning
        if (!decoder.ReachedFinal())
          KALDI_WARN << "Decoder did not reach end-state, "
                     << "outputting partial traceback since --allow-partial=true";
        num_success++;

        //解碼路徑tid
        std::vector<int32> alignment;
        //輸出的詞
        std::vector<int32> words;
        //解碼之後的代價
        LatticeWeight weight;

        //總幀數
        frame_count += features.NumRows();

        //得到lattice
        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

        //生成文件
        words_writer.Write(utt, words);
        //判斷是否有傳入參數(5) alignment_wspecifier，有傳入就寫文件
        if (alignment_wspecifier != "")
          alignment_writer.Write(utt, alignment);
        // //判斷是否有傳入參數(6) lattice_wspecifier
        if (lattice_wspecifier != "") {
          // We'll write the lattice without acoustic scaling.
          if (acoustic_scale != 0.0)
            fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                              &decoded);
          fst::VectorFst<CompactLatticeArc> clat;
          ConvertLattice(decoded, &clat, true);
          clat_writer.Write(utt, clat);
        }

        //有傳入就自動轉字 word-id -> word 
        if (word_syms != NULL) {
          std::cerr << utt << ' ';
          for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }
        BaseFloat like = -ConvertToCost(weight);
        tot_like += like;
        KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
                  << (like / features.NumRows()) << " over "
                  << features.NumRows() << " frames.";
      } else {
        num_fail++;
        KALDI_WARN << "Did not successfully decode utterance " << utt
                   << ", len = " << features.NumRows();
      }
    }

    //
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";

    delete word_syms;
    delete decode_fst;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


