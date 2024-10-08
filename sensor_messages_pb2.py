# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sensor_messages.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import state_messages_pb2 as state__messages__pb2
import map_messages_pb2 as map__messages__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='sensor_messages.proto',
  package='anantak.message',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x15sensor_messages.proto\x12\x0f\x61nantak.message\x1a\x14state_messages.proto\x1a\x12map_messages.proto\"_\n\tHeaderMsg\x12\x11\n\ttimestamp\x18\x01 \x02(\x03\x12\x0c\n\x04type\x18\x02 \x02(\t\x12\x19\n\x11recieve_timestamp\x18\x03 \x01(\x03\x12\x16\n\x0esend_timestamp\x18\x04 \x01(\x03\"\xa6\x01\n\x0cImageMessage\x12\x12\n\ncamera_num\x18\x65 \x02(\x05\x12\x0e\n\x06height\x18\x66 \x02(\x05\x12\r\n\x05width\x18g \x02(\x05\x12\r\n\x05\x64\x65pth\x18h \x02(\x05\x12\x13\n\nimage_data\x18\xc8\x01 \x02(\x0c\x12\x12\n\tposn_data\x18\xac\x02 \x03(\x02\x12\x14\n\x0b\x64\x65scription\x18\xde\x02 \x03(\x02\x12\x15\n\x0ctimestamp_us\x18\x90\x03 \x01(\x03\"\xc5\x06\n\tSensorMsg\x12*\n\x06header\x18\x01 \x02(\x0b\x32\x1a.anantak.message.HeaderMsg\x12\x12\n\nstring_msg\x18\x02 \x01(\t\x12\x30\n\timage_msg\x18\n \x01(\x0b\x32\x1d.anantak.message.ImageMessage\x12\x39\n\x0epose_state_msg\x18h \x01(\x0b\x32!.anantak.message.PoseStateMessage\x12\x43\n\x13pose_trajectory_msg\x18k \x01(\x0b\x32&.anantak.message.PoseTrajectoryMessage\x12\x38\n\rfreespace_msg\x18t \x01(\x0b\x32!.anantak.message.FreespaceMessage\x12\x36\n\x0clighting_msg\x18u \x01(\x0b\x32 .anantak.message.LightingMessage\x12\x30\n\troute_msg\x18v \x01(\x0b\x32\x1d.anantak.message.RouteMessage\x12=\n\x10line_segment_msg\x18x \x01(\x0b\x32#.anantak.message.LineSegmentMessage\x12?\n\x11line_segments_msg\x18y \x01(\x0b\x32$.anantak.message.LineSegmentsMessage\x12\x30\n\tpoint_msg\x18z \x01(\x0b\x32\x1d.anantak.message.PointMessage\x12\x32\n\npoints_msg\x18{ \x01(\x0b\x32\x1e.anantak.message.PointsMessage\x12\x32\n\nbeacon_msg\x18| \x01(\x0b\x32\x1e.anantak.message.BeaconMessage\x12\x36\n\x0csteering_msg\x18} \x01(\x0b\x32 .anantak.message.SteeringMessage\x12P\n\x1atraction_motor_encoder_msg\x18~ \x01(\x0b\x32,.anantak.message.TractionMotorEncoderMessage'
  ,
  dependencies=[state__messages__pb2.DESCRIPTOR,map__messages__pb2.DESCRIPTOR,])




_HEADERMSG = _descriptor.Descriptor(
  name='HeaderMsg',
  full_name='anantak.message.HeaderMsg',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='anantak.message.HeaderMsg.timestamp', index=0,
      number=1, type=3, cpp_type=2, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='anantak.message.HeaderMsg.type', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='recieve_timestamp', full_name='anantak.message.HeaderMsg.recieve_timestamp', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='send_timestamp', full_name='anantak.message.HeaderMsg.send_timestamp', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=84,
  serialized_end=179,
)


_IMAGEMESSAGE = _descriptor.Descriptor(
  name='ImageMessage',
  full_name='anantak.message.ImageMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='camera_num', full_name='anantak.message.ImageMessage.camera_num', index=0,
      number=101, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='height', full_name='anantak.message.ImageMessage.height', index=1,
      number=102, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width', full_name='anantak.message.ImageMessage.width', index=2,
      number=103, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='depth', full_name='anantak.message.ImageMessage.depth', index=3,
      number=104, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image_data', full_name='anantak.message.ImageMessage.image_data', index=4,
      number=200, type=12, cpp_type=9, label=2,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='posn_data', full_name='anantak.message.ImageMessage.posn_data', index=5,
      number=300, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='description', full_name='anantak.message.ImageMessage.description', index=6,
      number=350, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='timestamp_us', full_name='anantak.message.ImageMessage.timestamp_us', index=7,
      number=400, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=182,
  serialized_end=348,
)


_SENSORMSG = _descriptor.Descriptor(
  name='SensorMsg',
  full_name='anantak.message.SensorMsg',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='anantak.message.SensorMsg.header', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='string_msg', full_name='anantak.message.SensorMsg.string_msg', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image_msg', full_name='anantak.message.SensorMsg.image_msg', index=2,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='pose_state_msg', full_name='anantak.message.SensorMsg.pose_state_msg', index=3,
      number=104, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='pose_trajectory_msg', full_name='anantak.message.SensorMsg.pose_trajectory_msg', index=4,
      number=107, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='freespace_msg', full_name='anantak.message.SensorMsg.freespace_msg', index=5,
      number=116, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lighting_msg', full_name='anantak.message.SensorMsg.lighting_msg', index=6,
      number=117, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='route_msg', full_name='anantak.message.SensorMsg.route_msg', index=7,
      number=118, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='line_segment_msg', full_name='anantak.message.SensorMsg.line_segment_msg', index=8,
      number=120, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='line_segments_msg', full_name='anantak.message.SensorMsg.line_segments_msg', index=9,
      number=121, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='point_msg', full_name='anantak.message.SensorMsg.point_msg', index=10,
      number=122, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='points_msg', full_name='anantak.message.SensorMsg.points_msg', index=11,
      number=123, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='beacon_msg', full_name='anantak.message.SensorMsg.beacon_msg', index=12,
      number=124, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='steering_msg', full_name='anantak.message.SensorMsg.steering_msg', index=13,
      number=125, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='traction_motor_encoder_msg', full_name='anantak.message.SensorMsg.traction_motor_encoder_msg', index=14,
      number=126, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=351,
  serialized_end=1188,
)

_SENSORMSG.fields_by_name['header'].message_type = _HEADERMSG
_SENSORMSG.fields_by_name['image_msg'].message_type = _IMAGEMESSAGE
_SENSORMSG.fields_by_name['pose_state_msg'].message_type = state__messages__pb2._POSESTATEMESSAGE
_SENSORMSG.fields_by_name['pose_trajectory_msg'].message_type = state__messages__pb2._POSETRAJECTORYMESSAGE
_SENSORMSG.fields_by_name['freespace_msg'].message_type = state__messages__pb2._FREESPACEMESSAGE
_SENSORMSG.fields_by_name['lighting_msg'].message_type = state__messages__pb2._LIGHTINGMESSAGE
_SENSORMSG.fields_by_name['route_msg'].message_type = state__messages__pb2._ROUTEMESSAGE
_SENSORMSG.fields_by_name['line_segment_msg'].message_type = state__messages__pb2._LINESEGMENTMESSAGE
_SENSORMSG.fields_by_name['line_segments_msg'].message_type = state__messages__pb2._LINESEGMENTSMESSAGE
_SENSORMSG.fields_by_name['point_msg'].message_type = state__messages__pb2._POINTMESSAGE
_SENSORMSG.fields_by_name['points_msg'].message_type = state__messages__pb2._POINTSMESSAGE
_SENSORMSG.fields_by_name['beacon_msg'].message_type = state__messages__pb2._BEACONMESSAGE
_SENSORMSG.fields_by_name['steering_msg'].message_type = state__messages__pb2._STEERINGMESSAGE
_SENSORMSG.fields_by_name['traction_motor_encoder_msg'].message_type = state__messages__pb2._TRACTIONMOTORENCODERMESSAGE
DESCRIPTOR.message_types_by_name['HeaderMsg'] = _HEADERMSG
DESCRIPTOR.message_types_by_name['ImageMessage'] = _IMAGEMESSAGE
DESCRIPTOR.message_types_by_name['SensorMsg'] = _SENSORMSG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HeaderMsg = _reflection.GeneratedProtocolMessageType('HeaderMsg', (_message.Message,), {
  'DESCRIPTOR' : _HEADERMSG,
  '__module__' : 'sensor_messages_pb2'
  # @@protoc_insertion_point(class_scope:anantak.message.HeaderMsg)
  })
_sym_db.RegisterMessage(HeaderMsg)

ImageMessage = _reflection.GeneratedProtocolMessageType('ImageMessage', (_message.Message,), {
  'DESCRIPTOR' : _IMAGEMESSAGE,
  '__module__' : 'sensor_messages_pb2'
  # @@protoc_insertion_point(class_scope:anantak.message.ImageMessage)
  })
_sym_db.RegisterMessage(ImageMessage)

SensorMsg = _reflection.GeneratedProtocolMessageType('SensorMsg', (_message.Message,), {
  'DESCRIPTOR' : _SENSORMSG,
  '__module__' : 'sensor_messages_pb2'
  # @@protoc_insertion_point(class_scope:anantak.message.SensorMsg)
  })
_sym_db.RegisterMessage(SensorMsg)


# @@protoc_insertion_point(module_scope)
