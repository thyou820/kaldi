// decoder/simple-decoder.cc

// Copyright 2009-2011 Microsoft Corporation
//           2012-2013 Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/simple-decoder.h"
#include "fstext/remove-eps-local.h"
#include <algorithm>

namespace kaldi {

SimpleDecoder::~SimpleDecoder() {
  ClearToks(cur_toks_);
  ClearToks(prev_toks_);
}

// 解碼核心
bool SimpleDecoder::Decode(DecodableInterface *decodable) {
  // 初始化
  InitDecoding();

  // 解碼
  AdvanceDecoding(decodable);
  return (!cur_toks_.empty());
}

void SimpleDecoder::InitDecoding() {
  // clean up from last time:
  // 清空cur_toks
  ClearToks(cur_toks_);
  // 清空prev_toks_
  ClearToks(prev_toks_);
  
  // initialize decoding:
  // start_state : HCLG 的加粗圈
  StateId start_state = fst_.Start();
  
  KALDI_ASSERT(start_state != fst::kNoStateId);
  StdArc dummy_arc(0, 0, StdWeight::One(), start_state);
  // NULL:沒有*prev
  cur_toks_[start_state] = new Token(dummy_arc, 0.0, NULL);
  // 初始化解碼幀數
  num_frames_decoded_ = 0;
  // 重點
  ProcessNonemitting();
}
// *decodable : 聲學模型
void SimpleDecoder::AdvanceDecoding(DecodableInterface *decodable,
                                      int32 max_num_frames) {
  // 待解碼的音頻帧數要大於等於0
  KALDI_ASSERT(num_frames_decoded_ >= 0 &&
               "You must call InitDecoding() before AdvanceDecoding()");


  // 音頻的帧數
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  // num_frames_ready : 音頻一共的帧數
  // num_frames_decoded_ : 當前解碼的帧數
  // 保證音頻的帧數 > 當前解碼的帧數
  KALDI_ASSERT(num_frames_ready >= num_frames_decoded_);
  int32 target_frames_decoded = num_frames_ready;
  // simple-decoder.h 裡 max_num_frames有默認值 = -1
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     num_frames_decoded_ + max_num_frames);
  // ***核心
  // 一帧一帧解碼(如果還有帧,則繼續解碼)
  while (num_frames_decoded_ < target_frames_decoded) {
    // note: ProcessEmitting() increments num_frames_decoded_
    // 清除了prev_toks
    ClearToks(prev_toks_);
    // 交換toks
    cur_toks_.swap(prev_toks_);
    // 拓展實邊
    ProcessEmitting(decodable);
    // 拓展虛邊
    ProcessNonemitting();
    // 處理令牌(減枝)
    PruneToks(beam_, &cur_toks_);
  }
}

bool SimpleDecoder::ReachedFinal() const {
  for (unordered_map<StateId, Token*>::const_iterator iter = cur_toks_.begin();
       iter != cur_toks_.end();
       ++iter) {
    if (iter->second->cost_ != std::numeric_limits<BaseFloat>::infinity() &&
        fst_.Final(iter->first) != StdWeight::Zero())
      return true;
  }
  return false;
}

BaseFloat SimpleDecoder::FinalRelativeCost() const {
  // as a special case, if there are no active tokens at all (e.g. some kind of
  // pruning failure), return infinity.
  double infinity = std::numeric_limits<double>::infinity();
  if (cur_toks_.empty())
    return infinity;
  double best_cost = infinity,
      best_cost_with_final = infinity;
  for (unordered_map<StateId, Token*>::const_iterator iter = cur_toks_.begin();
       iter != cur_toks_.end();
       ++iter) {
    // Note: Plus is taking the minimum cost, since we're in the tropical
    // semiring.
    best_cost = std::min(best_cost, iter->second->cost_);
    best_cost_with_final = std::min(best_cost_with_final,
                                    iter->second->cost_ +
                                    fst_.Final(iter->first).Value());
  }
  BaseFloat extra_cost = best_cost_with_final - best_cost;
  if (extra_cost != extra_cost) { // NaN.  This shouldn't happen; it indicates some
                                  // kind of error, most likely.
    KALDI_WARN << "Found NaN (likely search failure in decoding)";
    return infinity;
  }
  // Note: extra_cost will be infinity if no states were final.
  return extra_cost;
}

// Outputs an FST corresponding to the single best path
// through the lattice.
bool SimpleDecoder::GetBestPath(Lattice *fst_out, bool use_final_probs) const {
  fst_out->DeleteStates();
  Token *best_tok = NULL;
  bool is_final = ReachedFinal();
  if (!is_final) {
    for (unordered_map<StateId, Token*>::const_iterator iter = cur_toks_.begin();
         iter != cur_toks_.end();
         ++iter)
      if (best_tok == NULL || *best_tok < *(iter->second) )
        best_tok = iter->second;
  } else {
    double infinity =std::numeric_limits<double>::infinity(),
        best_cost = infinity;
    for (unordered_map<StateId, Token*>::const_iterator iter = cur_toks_.begin();
         iter != cur_toks_.end();
         ++iter) {
      double this_cost = iter->second->cost_ + fst_.Final(iter->first).Value();
      if (this_cost != infinity && this_cost < best_cost) {
        best_cost = this_cost;
        best_tok = iter->second;
      }
    }
  }
  if (best_tok == NULL) return false;  // No output.

  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.
  for (Token *tok = best_tok; tok != NULL; tok = tok->prev_)
    arcs_reverse.push_back(tok->arc_);
  KALDI_ASSERT(arcs_reverse.back().nextstate == fst_.Start());
  arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

  StateId cur_state = fst_out->AddState();
  fst_out->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = fst_out->AddState();
    fst_out->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final && use_final_probs)
    fst_out->SetFinal(cur_state,
                      LatticeWeight(fst_.Final(best_tok->arc_.nextstate).Value(),
                                    0.0));
  else
    fst_out->SetFinal(cur_state, LatticeWeight::One());
  fst::RemoveEpsLocal(fst_out);
  return true;
}


void SimpleDecoder::ProcessEmitting(DecodableInterface *decodable) {

  // 當前解碼的帧數
  int32 frame = num_frames_decoded_;
  // Processes emitting arcs for one frame.  Propagates from
  // prev_toks_ to cur_toks_.
  // 代價,初始為正無窮
  double cutoff = std::numeric_limits<BaseFloat>::infinity();
  // 遍歷prev裡面的節點
  for (unordered_map<StateId, Token*>::iterator iter = prev_toks_.begin();
       iter != prev_toks_.end();
       ++iter) {
    // 得到的節點
    StateId state = iter->first;
    // 得到節點上的Token 
    Token *tok = iter->second;
    // 保證State和弧上的節點保持一致
    KALDI_ASSERT(state == tok->arc_.nextstate);
    
    // 遍歷從state節點中出發的弧
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      // 遍歷的每一條弧
      const StdArc &arc = aiter.Value();
      // std::cout<<"遍歷的節點 : " << state <<"", 第: "<< frame << "幀, ilable: " << arc.ilable << std::endl;
      if (arc.ilabel != 0) {  // propagate..

        // 得到聲學模型的代價
        BaseFloat acoustic_cost = -decodable->LogLikelihood(frame, arc.ilabel);
        // std::cout<< "聲學模型代價: " << acoustic_cost << std::endl;
        double total_cost = tok->cost_ + arc.weight.Value() + acoustic_cost;

        // cutoff初始是正無窮
        if (total_cost >= cutoff) continue;
        if (total_cost + beam_  < cutoff)
          cutoff = total_cost + beam_;
        
        // 新建一個token (ark 走的哪條邊,聲學模型的代價,上一個Tokem)  
        Token *new_tok = new Token(arc, acoustic_cost, tok);
        
        // 查找ark邊上的節點是否有令牌
        unordered_map<StateId, Token*>::iterator find_iter
            = cur_toks_.find(arc.nextstate);

        // 表示沒有找到
        if (find_iter == cur_toks_.end()) {
          // 賦值節點上的令牌
          cur_toks_[arc.nextstate] = new_tok;
          // std::cout<< "添加的令牌: " << arc.nextstate << " 節點 ->" << new_tok->cost_<< "代價" << std::endl; 
        } else {
          //如果這個節點上有令牌,對比哪個代價更小，刪掉代價大的令牌
          if ( *(find_iter->second) < *new_tok ) {
            Token::TokenDelete(find_iter->second);
            find_iter->second = new_tok;
          } else {
            Token::TokenDelete(new_tok);
          }
        }
      }
    }
  }
  num_frames_decoded_++;
}

// 拓展空邊
void SimpleDecoder::ProcessNonemitting() {
  // Processes nonemitting arcs for one frame.  Propagates within
  // cur_toks_.
  std::vector<StateId> queue;
  double infinity = std::numeric_limits<double>::infinity();
  // 宣告一個best_cost，並初始化為infinity
  double best_cost = infinity;
  // 遍歷cur_toks
  for (unordered_map<StateId, Token*>::iterator iter = cur_toks_.begin();
       iter != cur_toks_.end();
       ++iter) {
    // 把cur中的state id添加到queue中
    // first,second是c++的一種迭代方式，指向的是toks的第一個值與第二個值
    // 這裡的first是 state id，second是cost 
    queue.push_back(iter->first);
    // 取得代價比較小的
    // std::cout<<"iter->first :"<< iter->first << std::endl;
    // std::cout<<"best_cost :"<< best_cost << std::endl;
    // std::cout<<"iter->second->cost_ :"<< iter->second->cost_ << std::endl;
    best_cost = std::min(best_cost, iter->second->cost_);
  }
  // 剪枝代價
  // 減少枝葉(減枝)
  double cutoff = best_cost + beam_;
  // std::cout<<"cutoff_after :"<< cutoff << std::endl;
  while (!queue.empty()) {
    StateId state = queue.back();
    // std::cout<<"pop_state_id: "<< state << std::endl;
    queue.pop_back();
    Token *tok = cur_toks_[state];
    // 安全校驗
    KALDI_ASSERT(tok != NULL && state == tok->arc_.nextstate);
    
    //遍歷HCLG
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const StdArc &arc = aiter.Value();
      // std::cout<<"-----ProcessNonemitting----- arc.ialabel : "<< arc.ialabel << std::endl;

      // 拓展空邊
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        const BaseFloat acoustic_cost = 0.0;
        Token *new_tok = new Token(arc, acoustic_cost, tok);
        if (new_tok->cost_ > cutoff) {
          Token::TokenDelete(new_tok);
        } else {
          unordered_map<StateId, Token*>::iterator find_iter
              = cur_toks_.find(arc.nextstate);
          if (find_iter == cur_toks_.end()) {
            cur_toks_[arc.nextstate] = new_tok;
            queue.push_back(arc.nextstate);
          } else {
            if ( *(find_iter->second) < *new_tok ) {
              Token::TokenDelete(find_iter->second);
              find_iter->second = new_tok;
              queue.push_back(arc.nextstate);
            } else {
              Token::TokenDelete(new_tok);
            }
          }
        }
      }
    }
  }
}

// static
void SimpleDecoder::ClearToks(unordered_map<StateId, Token*> &toks) {
  for (unordered_map<StateId, Token*>::iterator iter = toks.begin();
       iter != toks.end(); ++iter) {
    Token::TokenDelete(iter->second);
  }
  toks.clear();
}

// static
void SimpleDecoder::PruneToks(BaseFloat beam, unordered_map<StateId, Token*> *toks) {
  if (toks->empty()) {
    KALDI_VLOG(2) <<  "No tokens to prune.\n";
    return;
  }

  double best_cost = std::numeric_limits<double>::infinity();

  // 遍歷cur_toknes 
  for (unordered_map<StateId, Token*>::iterator iter = toks->begin();
       iter != toks->end(); ++iter)
    // 找到一個最小的代價
    best_cost = std::min(best_cost, iter->second->cost_);

  std::vector<StateId> retained;
  double cutoff = best_cost + beam;

  // 遍歷cur_toknes 
  for (unordered_map<StateId, Token*>::iterator iter = toks->begin();
       iter != toks->end(); ++iter) {
    // 如果遍歷的代價比cutoff小
    if (iter->second->cost_ < cutoff)
      // 保留stateid 
      retained.push_back(iter->first);
    else
      // 否則刪除
      Token::TokenDelete(iter->second);
  }

  unordered_map<StateId, Token*> tmp;
  // 遍歷保留了那些stateid 
  for (size_t i = 0; i < retained.size(); i++) {
    // 把retained中的stateid所對應的Token賦值給tmp中的state id
    tmp[retained[i]] = (*toks)[retained[i]];
  }
  KALDI_VLOG(2) <<  "Pruned to " << (retained.size()) << " toks.\n";
  // 剪枝 : 把cur_toks 換成 tmp   
  tmp.swap(*toks);
}

} // end namespace kaldi.
