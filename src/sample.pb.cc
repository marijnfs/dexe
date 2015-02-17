// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: sample.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "sample.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace {

const ::google::protobuf::Descriptor* Sample_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  Sample_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_sample_2eproto() {
  protobuf_AddDesc_sample_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "sample.proto");
  GOOGLE_CHECK(file != NULL);
  Sample_descriptor_ = file->message_type(0);
  static const int Sample_offsets_[2] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, input_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, output_),
  };
  Sample_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      Sample_descriptor_,
      Sample::default_instance_,
      Sample_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Sample, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(Sample));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_sample_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    Sample_descriptor_, &Sample::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_sample_2eproto() {
  delete Sample::default_instance_;
  delete Sample_reflection_;
}

void protobuf_AddDesc_sample_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\014sample.proto\"/\n\006Sample\022\021\n\005input\030\001 \003(\002B"
    "\002\020\001\022\022\n\006output\030\002 \003(\002B\002\020\001", 63);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "sample.proto", &protobuf_RegisterTypes);
  Sample::default_instance_ = new Sample();
  Sample::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_sample_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_sample_2eproto {
  StaticDescriptorInitializer_sample_2eproto() {
    protobuf_AddDesc_sample_2eproto();
  }
} static_descriptor_initializer_sample_2eproto_;

// ===================================================================

#ifndef _MSC_VER
const int Sample::kInputFieldNumber;
const int Sample::kOutputFieldNumber;
#endif  // !_MSC_VER

Sample::Sample()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void Sample::InitAsDefaultInstance() {
}

Sample::Sample(const Sample& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void Sample::SharedCtor() {
  _cached_size_ = 0;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

Sample::~Sample() {
  SharedDtor();
}

void Sample::SharedDtor() {
  if (this != default_instance_) {
  }
}

void Sample::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Sample::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return Sample_descriptor_;
}

const Sample& Sample::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_sample_2eproto();
  return *default_instance_;
}

Sample* Sample::default_instance_ = NULL;

Sample* Sample::New() const {
  return new Sample;
}

void Sample::Clear() {
  input_.Clear();
  output_.Clear();
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool Sample::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated float input = 1 [packed = true];
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_input())));
        } else if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag)
                   == ::google::protobuf::internal::WireFormatLite::
                      WIRETYPE_FIXED32) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 10, input, this->mutable_input())));
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(18)) goto parse_output;
        break;
      }

      // repeated float output = 2 [packed = true];
      case 2: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
         parse_output:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_output())));
        } else if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag)
                   == ::google::protobuf::internal::WireFormatLite::
                      WIRETYPE_FIXED32) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 18, input, this->mutable_output())));
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectAtEnd()) return true;
        break;
      }

      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void Sample::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // repeated float input = 1 [packed = true];
  if (this->input_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(1, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_input_cached_byte_size_);
  }
  for (int i = 0; i < this->input_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteFloatNoTag(
      this->input(i), output);
  }

  // repeated float output = 2 [packed = true];
  if (this->output_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(2, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_output_cached_byte_size_);
  }
  for (int i = 0; i < this->output_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteFloatNoTag(
      this->output(i), output);
  }

  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* Sample::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // repeated float input = 1 [packed = true];
  if (this->input_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      1,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
      _input_cached_byte_size_, target);
  }
  for (int i = 0; i < this->input_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteFloatNoTagToArray(this->input(i), target);
  }

  // repeated float output = 2 [packed = true];
  if (this->output_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      2,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
      _output_cached_byte_size_, target);
  }
  for (int i = 0; i < this->output_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteFloatNoTagToArray(this->output(i), target);
  }

  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int Sample::ByteSize() const {
  int total_size = 0;

  // repeated float input = 1 [packed = true];
  {
    int data_size = 0;
    data_size = 4 * this->input_size();
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(data_size);
    }
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _input_cached_byte_size_ = data_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // repeated float output = 2 [packed = true];
  {
    int data_size = 0;
    data_size = 4 * this->output_size();
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(data_size);
    }
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _output_cached_byte_size_ = data_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Sample::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const Sample* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const Sample*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void Sample::MergeFrom(const Sample& from) {
  GOOGLE_CHECK_NE(&from, this);
  input_.MergeFrom(from.input_);
  output_.MergeFrom(from.output_);
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void Sample::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Sample::CopyFrom(const Sample& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Sample::IsInitialized() const {

  return true;
}

void Sample::Swap(Sample* other) {
  if (other != this) {
    input_.Swap(&other->input_);
    output_.Swap(&other->output_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata Sample::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = Sample_descriptor_;
  metadata.reflection = Sample_reflection_;
  return metadata;
}


// @@protoc_insertion_point(namespace_scope)

// @@protoc_insertion_point(global_scope)
