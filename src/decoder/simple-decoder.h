// decoder/simple-decoder.h

// Copyright 2009-2013  Microsoft Corporation;  Lukas Burget;
//                      Saarland University (author: Arnab Ghoshal);
//                      Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_DECODER_SIMPLE_DECODER_H_
#define KALDI_DECODER_SIMPLE_DECODER_H_


#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"

namespace kaldi {

/** Simplest possible decoder, included largely for didactic purposes and as a
    means to debug more highly optimized decoders.  See \ref decoders_simple
    for more information.
 */
class SimpleDecoder {
 public:
  // 邊 (輸入,輸出.代價)
  typedef fst::StdArc StdArc;
  // 代價
  typedef StdArc::Weight StdWeight;
  // 輸出
  typedef StdArc::Label Label;
  // 
  typedef StdArc::StateId StateId;

  //構造方法
  SimpleDecoder(const fst::Fst<fst::StdArc> &fst, BaseFloat beam): fst_(fst), beam_(beam) { }

  //析構函數
  ~SimpleDecoder();

  /// Decode this utterance.
  /// Returns true if any tokens reached the end of the file (regardless of
  /// whether they are in a final state); query ReachedFinal() after Decode()
  /// to see whether we reached a final state.
  /// 解碼一段語音
  bool Decode(DecodableInterface *decodable);

  //判斷有沒有到中止節點
  bool ReachedFinal() const;

  // GetBestPath gets the decoding traceback. If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into account final-probs.
  // fst_out will be empty (Start() == kNoStateId) if nothing was available due to
  // search error.
  // If Decode() returned true, it is safe to assume GetBestPath will return true.
  // It returns true if the output lattice was nonempty (i.e. had states in it);
  // using the return value is deprecated.
  // 獲取最優的輸出路徑
  bool GetBestPath(Lattice *fst_out, bool use_final_probs = true) const;

  /// *** The next functions are from the "new interface". ***

  /// FinalRelativeCost() serves the same function as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost plus
  /// cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were present
  /// on the final frame.  It will usually be nonnegative.
  
  BaseFloat FinalRelativeCost() const;

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need
  /// to call this.  You can call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance.
  /// 初始化解碼器，每一段語音進來之後，都要初始化解碼器
  void InitDecoding();

  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.  If it returns false, then no tokens are alive,
  /// which is a kind of error state.
  /// 解碼一段語音
  void AdvanceDecoding(DecodableInterface *decodable,
                         int32 max_num_frames = -1);

  /// Returns the number of frames already decoded.
  /// 已經解碼多少幀語音
  int32 NumFramesDecoded() const { return num_frames_decoded_; }

 private:
  // 令牌
  class Token {
   public:
    // arc 
    LatticeArc arc_; // We use LatticeArc so that we can separately
                     // store the acoustic and graph cost, in case
                     // we need to produce lattice-formatted output.
    // 前向指針,回溯最優路徑
    Token *prev_;
    // 前面有多少個節點
    int32 ref_count_;
    // 代價
    double cost_; // accumulated total cost up to this point.
    Token(const StdArc &arc,
          BaseFloat acoustic_cost,
          Token *prev): prev_(prev), ref_count_(1) {
      // 輸入標籤
      arc_.ilabel = arc.ilabel;
      // 輸出標籤
      arc_.olabel = arc.olabel;
      // 代價
      arc_.weight = LatticeWeight(arc.weight.Value(), acoustic_cost);
      // 邊指向的下一個狀態
      arc_.nextstate = arc.nextstate;
      if (prev) {
        // 之前節點++
        prev->ref_count_++;
        // 代價到該節點的總代價
        cost_ = prev->cost_ + (arc.weight.Value() + acoustic_cost);
      } else {
        cost_ = arc.weight.Value() + acoustic_cost;
      }
    }
    // 運算符重載
    bool operator < (const Token &other) {
      return cost_ > other.cost_;
    }

    static void TokenDelete(Token *tok) {
      while (--tok->ref_count_ == 0) {
        Token *prev = tok->prev_;
        delete tok;
        if (prev == NULL) return;
        else tok = prev;
      }
#ifdef KALDI_PARANOID
      KALDI_ASSERT(tok->ref_count_ > 0);
#endif
    }
  };

  // ProcessEmitting decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  // 拓展實邊 (trasistion-id 不等於0的)
  void ProcessEmitting(DecodableInterface *decodable);

  // 拓展空邊 (trasistion-id 等於0的 ， 特殊的邊)
  void ProcessNonemitting();

  // 當前解碼的Token
  unordered_map<StateId, Token*> cur_toks_;
  // 上一幀解碼的Token
  unordered_map<StateId, Token*> prev_toks_;
  // HCLG
  const fst::Fst<fst::StdArc> &fst_;
  BaseFloat beam_;
  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  static void ClearToks(unordered_map<StateId, Token*> &toks);

  static void PruneToks(BaseFloat beam, unordered_map<StateId, Token*> *toks);

  KALDI_DISALLOW_COPY_AND_ASSIGN(SimpleDecoder);
};


} // end namespace kaldi.


#endif
