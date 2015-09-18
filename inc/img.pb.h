// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: img.proto

#ifndef PROTOBUF_img_2eproto__INCLUDED
#define PROTOBUF_img_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2005000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2005000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_img_2eproto();
void protobuf_AssignDesc_img_2eproto();
void protobuf_ShutdownFile_img_2eproto();

class Img;

// ===================================================================

class Img : public ::google::protobuf::Message {
 public:
  Img();
  virtual ~Img();

  Img(const Img& from);

  inline Img& operator=(const Img& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Img& default_instance();

  void Swap(Img* other);

  // implements Message ----------------------------------------------

  Img* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const Img& from);
  void MergeFrom(const Img& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required uint32 w = 1;
  inline bool has_w() const;
  inline void clear_w();
  static const int kWFieldNumber = 1;
  inline ::google::protobuf::uint32 w() const;
  inline void set_w(::google::protobuf::uint32 value);

  // required uint32 h = 2;
  inline bool has_h() const;
  inline void clear_h();
  static const int kHFieldNumber = 2;
  inline ::google::protobuf::uint32 h() const;
  inline void set_h(::google::protobuf::uint32 value);

  // required uint32 c = 3;
  inline bool has_c() const;
  inline void clear_c();
  static const int kCFieldNumber = 3;
  inline ::google::protobuf::uint32 c() const;
  inline void set_c(::google::protobuf::uint32 value);

  // required float time = 4;
  inline bool has_time() const;
  inline void clear_time();
  static const int kTimeFieldNumber = 4;
  inline float time() const;
  inline void set_time(float value);

  // repeated float data = 5 [packed = true];
  inline int data_size() const;
  inline void clear_data();
  static const int kDataFieldNumber = 5;
  inline float data(int index) const;
  inline void set_data(int index, float value);
  inline void add_data(float value);
  inline const ::google::protobuf::RepeatedField< float >&
      data() const;
  inline ::google::protobuf::RepeatedField< float >*
      mutable_data();

  // repeated float state = 6 [packed = true];
  inline int state_size() const;
  inline void clear_state();
  static const int kStateFieldNumber = 6;
  inline float state(int index) const;
  inline void set_state(int index, float value);
  inline void add_state(float value);
  inline const ::google::protobuf::RepeatedField< float >&
      state() const;
  inline ::google::protobuf::RepeatedField< float >*
      mutable_state();

  // repeated float action = 7 [packed = true];
  inline int action_size() const;
  inline void clear_action();
  static const int kActionFieldNumber = 7;
  inline float action(int index) const;
  inline void set_action(int index, float value);
  inline void add_action(float value);
  inline const ::google::protobuf::RepeatedField< float >&
      action() const;
  inline ::google::protobuf::RepeatedField< float >*
      mutable_action();

  // @@protoc_insertion_point(class_scope:Img)
 private:
  inline void set_has_w();
  inline void clear_has_w();
  inline void set_has_h();
  inline void clear_has_h();
  inline void set_has_c();
  inline void clear_has_c();
  inline void set_has_time();
  inline void clear_has_time();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 w_;
  ::google::protobuf::uint32 h_;
  ::google::protobuf::uint32 c_;
  float time_;
  ::google::protobuf::RepeatedField< float > data_;
  mutable int _data_cached_byte_size_;
  ::google::protobuf::RepeatedField< float > state_;
  mutable int _state_cached_byte_size_;
  ::google::protobuf::RepeatedField< float > action_;
  mutable int _action_cached_byte_size_;

  mutable int _cached_size_;
  ::google::protobuf::uint32 _has_bits_[(7 + 31) / 32];

  friend void  protobuf_AddDesc_img_2eproto();
  friend void protobuf_AssignDesc_img_2eproto();
  friend void protobuf_ShutdownFile_img_2eproto();

  void InitAsDefaultInstance();
  static Img* default_instance_;
};
// ===================================================================


// ===================================================================

// Img

// required uint32 w = 1;
inline bool Img::has_w() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void Img::set_has_w() {
  _has_bits_[0] |= 0x00000001u;
}
inline void Img::clear_has_w() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void Img::clear_w() {
  w_ = 0u;
  clear_has_w();
}
inline ::google::protobuf::uint32 Img::w() const {
  return w_;
}
inline void Img::set_w(::google::protobuf::uint32 value) {
  set_has_w();
  w_ = value;
}

// required uint32 h = 2;
inline bool Img::has_h() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void Img::set_has_h() {
  _has_bits_[0] |= 0x00000002u;
}
inline void Img::clear_has_h() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void Img::clear_h() {
  h_ = 0u;
  clear_has_h();
}
inline ::google::protobuf::uint32 Img::h() const {
  return h_;
}
inline void Img::set_h(::google::protobuf::uint32 value) {
  set_has_h();
  h_ = value;
}

// required uint32 c = 3;
inline bool Img::has_c() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void Img::set_has_c() {
  _has_bits_[0] |= 0x00000004u;
}
inline void Img::clear_has_c() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void Img::clear_c() {
  c_ = 0u;
  clear_has_c();
}
inline ::google::protobuf::uint32 Img::c() const {
  return c_;
}
inline void Img::set_c(::google::protobuf::uint32 value) {
  set_has_c();
  c_ = value;
}

// required float time = 4;
inline bool Img::has_time() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void Img::set_has_time() {
  _has_bits_[0] |= 0x00000008u;
}
inline void Img::clear_has_time() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void Img::clear_time() {
  time_ = 0;
  clear_has_time();
}
inline float Img::time() const {
  return time_;
}
inline void Img::set_time(float value) {
  set_has_time();
  time_ = value;
}

// repeated float data = 5 [packed = true];
inline int Img::data_size() const {
  return data_.size();
}
inline void Img::clear_data() {
  data_.Clear();
}
inline float Img::data(int index) const {
  return data_.Get(index);
}
inline void Img::set_data(int index, float value) {
  data_.Set(index, value);
}
inline void Img::add_data(float value) {
  data_.Add(value);
}
inline const ::google::protobuf::RepeatedField< float >&
Img::data() const {
  return data_;
}
inline ::google::protobuf::RepeatedField< float >*
Img::mutable_data() {
  return &data_;
}

// repeated float state = 6 [packed = true];
inline int Img::state_size() const {
  return state_.size();
}
inline void Img::clear_state() {
  state_.Clear();
}
inline float Img::state(int index) const {
  return state_.Get(index);
}
inline void Img::set_state(int index, float value) {
  state_.Set(index, value);
}
inline void Img::add_state(float value) {
  state_.Add(value);
}
inline const ::google::protobuf::RepeatedField< float >&
Img::state() const {
  return state_;
}
inline ::google::protobuf::RepeatedField< float >*
Img::mutable_state() {
  return &state_;
}

// repeated float action = 7 [packed = true];
inline int Img::action_size() const {
  return action_.size();
}
inline void Img::clear_action() {
  action_.Clear();
}
inline float Img::action(int index) const {
  return action_.Get(index);
}
inline void Img::set_action(int index, float value) {
  action_.Set(index, value);
}
inline void Img::add_action(float value) {
  action_.Add(value);
}
inline const ::google::protobuf::RepeatedField< float >&
Img::action() const {
  return action_;
}
inline ::google::protobuf::RepeatedField< float >*
Img::mutable_action() {
  return &action_;
}


// @@protoc_insertion_point(namespace_scope)

#ifndef SWIG
namespace google {
namespace protobuf {


}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_img_2eproto__INCLUDED
