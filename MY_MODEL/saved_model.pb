ъљ)
Я░
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Щ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%иЛ8"&
exponential_avg_factorfloat%  ђ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ѓ
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Нб!
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
ю
module_wrapper/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namemodule_wrapper/conv2d/kernel
Ћ
0module_wrapper/conv2d/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper/conv2d/kernel*&
_output_shapes
: *
dtype0
ї
module_wrapper/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namemodule_wrapper/conv2d/bias
Ё
.module_wrapper/conv2d/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper/conv2d/bias*
_output_shapes
: *
dtype0
г
*module_wrapper_2/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*module_wrapper_2/batch_normalization/gamma
Ц
>module_wrapper_2/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp*module_wrapper_2/batch_normalization/gamma*
_output_shapes
: *
dtype0
ф
)module_wrapper_2/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)module_wrapper_2/batch_normalization/beta
Б
=module_wrapper_2/batch_normalization/beta/Read/ReadVariableOpReadVariableOp)module_wrapper_2/batch_normalization/beta*
_output_shapes
: *
dtype0
И
0module_wrapper_2/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20module_wrapper_2/batch_normalization/moving_mean
▒
Dmodule_wrapper_2/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp0module_wrapper_2/batch_normalization/moving_mean*
_output_shapes
: *
dtype0
└
4module_wrapper_2/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64module_wrapper_2/batch_normalization/moving_variance
╣
Hmodule_wrapper_2/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp4module_wrapper_2/batch_normalization/moving_variance*
_output_shapes
: *
dtype0
ц
 module_wrapper_4/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*1
shared_name" module_wrapper_4/conv2d_1/kernel
Ю
4module_wrapper_4/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_4/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
ћ
module_wrapper_4/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name module_wrapper_4/conv2d_1/bias
Ї
2module_wrapper_4/conv2d_1/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_4/conv2d_1/bias*
_output_shapes
:@*
dtype0
░
,module_wrapper_6/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,module_wrapper_6/batch_normalization_1/gamma
Е
@module_wrapper_6/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp,module_wrapper_6/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
«
+module_wrapper_6/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+module_wrapper_6/batch_normalization_1/beta
Д
?module_wrapper_6/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp+module_wrapper_6/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
╝
2module_wrapper_6/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42module_wrapper_6/batch_normalization_1/moving_mean
х
Fmodule_wrapper_6/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp2module_wrapper_6/batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
─
6module_wrapper_6/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*G
shared_name86module_wrapper_6/batch_normalization_1/moving_variance
й
Jmodule_wrapper_6/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp6module_wrapper_6/batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
ц
 module_wrapper_8/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*1
shared_name" module_wrapper_8/conv2d_2/kernel
Ю
4module_wrapper_8/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_8/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0
ћ
module_wrapper_8/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name module_wrapper_8/conv2d_2/bias
Ї
2module_wrapper_8/conv2d_2/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_8/conv2d_2/bias*
_output_shapes
:@*
dtype0
▓
-module_wrapper_10/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-module_wrapper_10/batch_normalization_2/gamma
Ф
Amodule_wrapper_10/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp-module_wrapper_10/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
░
,module_wrapper_10/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,module_wrapper_10/batch_normalization_2/beta
Е
@module_wrapper_10/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp,module_wrapper_10/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
Й
3module_wrapper_10/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53module_wrapper_10/batch_normalization_2/moving_mean
и
Gmodule_wrapper_10/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp3module_wrapper_10/batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
к
7module_wrapper_10/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97module_wrapper_10/batch_normalization_2/moving_variance
┐
Kmodule_wrapper_10/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp7module_wrapper_10/batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
Д
!module_wrapper_12/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*2
shared_name#!module_wrapper_12/conv2d_3/kernel
а
5module_wrapper_12/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp!module_wrapper_12/conv2d_3/kernel*'
_output_shapes
:@ђ*
dtype0
Ќ
module_wrapper_12/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*0
shared_name!module_wrapper_12/conv2d_3/bias
љ
3module_wrapper_12/conv2d_3/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_12/conv2d_3/bias*
_output_shapes	
:ђ*
dtype0
│
-module_wrapper_14/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*>
shared_name/-module_wrapper_14/batch_normalization_3/gamma
г
Amodule_wrapper_14/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp-module_wrapper_14/batch_normalization_3/gamma*
_output_shapes	
:ђ*
dtype0
▒
,module_wrapper_14/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*=
shared_name.,module_wrapper_14/batch_normalization_3/beta
ф
@module_wrapper_14/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp,module_wrapper_14/batch_normalization_3/beta*
_output_shapes	
:ђ*
dtype0
┐
3module_wrapper_14/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*D
shared_name53module_wrapper_14/batch_normalization_3/moving_mean
И
Gmodule_wrapper_14/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp3module_wrapper_14/batch_normalization_3/moving_mean*
_output_shapes	
:ђ*
dtype0
К
7module_wrapper_14/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*H
shared_name97module_wrapper_14/batch_normalization_3/moving_variance
└
Kmodule_wrapper_14/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp7module_wrapper_14/batch_normalization_3/moving_variance*
_output_shapes	
:ђ*
dtype0
џ
module_wrapper_18/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*/
shared_name module_wrapper_18/dense/kernel
Њ
2module_wrapper_18/dense/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper_18/dense/kernel* 
_output_shapes
:
ђђ*
dtype0
Љ
module_wrapper_18/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*-
shared_namemodule_wrapper_18/dense/bias
і
0module_wrapper_18/dense/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_18/dense/bias*
_output_shapes	
:ђ*
dtype0
│
-module_wrapper_20/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*>
shared_name/-module_wrapper_20/batch_normalization_4/gamma
г
Amodule_wrapper_20/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp-module_wrapper_20/batch_normalization_4/gamma*
_output_shapes	
:ђ*
dtype0
▒
,module_wrapper_20/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*=
shared_name.,module_wrapper_20/batch_normalization_4/beta
ф
@module_wrapper_20/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp,module_wrapper_20/batch_normalization_4/beta*
_output_shapes	
:ђ*
dtype0
┐
3module_wrapper_20/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*D
shared_name53module_wrapper_20/batch_normalization_4/moving_mean
И
Gmodule_wrapper_20/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp3module_wrapper_20/batch_normalization_4/moving_mean*
_output_shapes	
:ђ*
dtype0
К
7module_wrapper_20/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*H
shared_name97module_wrapper_20/batch_normalization_4/moving_variance
└
Kmodule_wrapper_20/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp7module_wrapper_20/batch_normalization_4/moving_variance*
_output_shapes	
:ђ*
dtype0
Ю
 module_wrapper_22/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*1
shared_name" module_wrapper_22/dense_1/kernel
ќ
4module_wrapper_22/dense_1/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_22/dense_1/kernel*
_output_shapes
:	ђ*
dtype0
ћ
module_wrapper_22/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name module_wrapper_22/dense_1/bias
Ї
2module_wrapper_22/dense_1/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_22/dense_1/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Х
)SGD/module_wrapper/conv2d/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)SGD/module_wrapper/conv2d/kernel/momentum
»
=SGD/module_wrapper/conv2d/kernel/momentum/Read/ReadVariableOpReadVariableOp)SGD/module_wrapper/conv2d/kernel/momentum*&
_output_shapes
: *
dtype0
д
'SGD/module_wrapper/conv2d/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/module_wrapper/conv2d/bias/momentum
Ъ
;SGD/module_wrapper/conv2d/bias/momentum/Read/ReadVariableOpReadVariableOp'SGD/module_wrapper/conv2d/bias/momentum*
_output_shapes
: *
dtype0
к
7SGD/module_wrapper_2/batch_normalization/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97SGD/module_wrapper_2/batch_normalization/gamma/momentum
┐
KSGD/module_wrapper_2/batch_normalization/gamma/momentum/Read/ReadVariableOpReadVariableOp7SGD/module_wrapper_2/batch_normalization/gamma/momentum*
_output_shapes
: *
dtype0
─
6SGD/module_wrapper_2/batch_normalization/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86SGD/module_wrapper_2/batch_normalization/beta/momentum
й
JSGD/module_wrapper_2/batch_normalization/beta/momentum/Read/ReadVariableOpReadVariableOp6SGD/module_wrapper_2/batch_normalization/beta/momentum*
_output_shapes
: *
dtype0
Й
-SGD/module_wrapper_4/conv2d_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*>
shared_name/-SGD/module_wrapper_4/conv2d_1/kernel/momentum
и
ASGD/module_wrapper_4/conv2d_1/kernel/momentum/Read/ReadVariableOpReadVariableOp-SGD/module_wrapper_4/conv2d_1/kernel/momentum*&
_output_shapes
: @*
dtype0
«
+SGD/module_wrapper_4/conv2d_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+SGD/module_wrapper_4/conv2d_1/bias/momentum
Д
?SGD/module_wrapper_4/conv2d_1/bias/momentum/Read/ReadVariableOpReadVariableOp+SGD/module_wrapper_4/conv2d_1/bias/momentum*
_output_shapes
:@*
dtype0
╩
9SGD/module_wrapper_6/batch_normalization_1/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*J
shared_name;9SGD/module_wrapper_6/batch_normalization_1/gamma/momentum
├
MSGD/module_wrapper_6/batch_normalization_1/gamma/momentum/Read/ReadVariableOpReadVariableOp9SGD/module_wrapper_6/batch_normalization_1/gamma/momentum*
_output_shapes
:@*
dtype0
╚
8SGD/module_wrapper_6/batch_normalization_1/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8SGD/module_wrapper_6/batch_normalization_1/beta/momentum
┴
LSGD/module_wrapper_6/batch_normalization_1/beta/momentum/Read/ReadVariableOpReadVariableOp8SGD/module_wrapper_6/batch_normalization_1/beta/momentum*
_output_shapes
:@*
dtype0
Й
-SGD/module_wrapper_8/conv2d_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*>
shared_name/-SGD/module_wrapper_8/conv2d_2/kernel/momentum
и
ASGD/module_wrapper_8/conv2d_2/kernel/momentum/Read/ReadVariableOpReadVariableOp-SGD/module_wrapper_8/conv2d_2/kernel/momentum*&
_output_shapes
:@@*
dtype0
«
+SGD/module_wrapper_8/conv2d_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+SGD/module_wrapper_8/conv2d_2/bias/momentum
Д
?SGD/module_wrapper_8/conv2d_2/bias/momentum/Read/ReadVariableOpReadVariableOp+SGD/module_wrapper_8/conv2d_2/bias/momentum*
_output_shapes
:@*
dtype0
╠
:SGD/module_wrapper_10/batch_normalization_2/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*K
shared_name<:SGD/module_wrapper_10/batch_normalization_2/gamma/momentum
┼
NSGD/module_wrapper_10/batch_normalization_2/gamma/momentum/Read/ReadVariableOpReadVariableOp:SGD/module_wrapper_10/batch_normalization_2/gamma/momentum*
_output_shapes
:@*
dtype0
╩
9SGD/module_wrapper_10/batch_normalization_2/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*J
shared_name;9SGD/module_wrapper_10/batch_normalization_2/beta/momentum
├
MSGD/module_wrapper_10/batch_normalization_2/beta/momentum/Read/ReadVariableOpReadVariableOp9SGD/module_wrapper_10/batch_normalization_2/beta/momentum*
_output_shapes
:@*
dtype0
┴
.SGD/module_wrapper_12/conv2d_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*?
shared_name0.SGD/module_wrapper_12/conv2d_3/kernel/momentum
║
BSGD/module_wrapper_12/conv2d_3/kernel/momentum/Read/ReadVariableOpReadVariableOp.SGD/module_wrapper_12/conv2d_3/kernel/momentum*'
_output_shapes
:@ђ*
dtype0
▒
,SGD/module_wrapper_12/conv2d_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*=
shared_name.,SGD/module_wrapper_12/conv2d_3/bias/momentum
ф
@SGD/module_wrapper_12/conv2d_3/bias/momentum/Read/ReadVariableOpReadVariableOp,SGD/module_wrapper_12/conv2d_3/bias/momentum*
_output_shapes	
:ђ*
dtype0
═
:SGD/module_wrapper_14/batch_normalization_3/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*K
shared_name<:SGD/module_wrapper_14/batch_normalization_3/gamma/momentum
к
NSGD/module_wrapper_14/batch_normalization_3/gamma/momentum/Read/ReadVariableOpReadVariableOp:SGD/module_wrapper_14/batch_normalization_3/gamma/momentum*
_output_shapes	
:ђ*
dtype0
╦
9SGD/module_wrapper_14/batch_normalization_3/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*J
shared_name;9SGD/module_wrapper_14/batch_normalization_3/beta/momentum
─
MSGD/module_wrapper_14/batch_normalization_3/beta/momentum/Read/ReadVariableOpReadVariableOp9SGD/module_wrapper_14/batch_normalization_3/beta/momentum*
_output_shapes	
:ђ*
dtype0
┤
+SGD/module_wrapper_18/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*<
shared_name-+SGD/module_wrapper_18/dense/kernel/momentum
Г
?SGD/module_wrapper_18/dense/kernel/momentum/Read/ReadVariableOpReadVariableOp+SGD/module_wrapper_18/dense/kernel/momentum* 
_output_shapes
:
ђђ*
dtype0
Ф
)SGD/module_wrapper_18/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*:
shared_name+)SGD/module_wrapper_18/dense/bias/momentum
ц
=SGD/module_wrapper_18/dense/bias/momentum/Read/ReadVariableOpReadVariableOp)SGD/module_wrapper_18/dense/bias/momentum*
_output_shapes	
:ђ*
dtype0
═
:SGD/module_wrapper_20/batch_normalization_4/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*K
shared_name<:SGD/module_wrapper_20/batch_normalization_4/gamma/momentum
к
NSGD/module_wrapper_20/batch_normalization_4/gamma/momentum/Read/ReadVariableOpReadVariableOp:SGD/module_wrapper_20/batch_normalization_4/gamma/momentum*
_output_shapes	
:ђ*
dtype0
╦
9SGD/module_wrapper_20/batch_normalization_4/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*J
shared_name;9SGD/module_wrapper_20/batch_normalization_4/beta/momentum
─
MSGD/module_wrapper_20/batch_normalization_4/beta/momentum/Read/ReadVariableOpReadVariableOp9SGD/module_wrapper_20/batch_normalization_4/beta/momentum*
_output_shapes	
:ђ*
dtype0
и
-SGD/module_wrapper_22/dense_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*>
shared_name/-SGD/module_wrapper_22/dense_1/kernel/momentum
░
ASGD/module_wrapper_22/dense_1/kernel/momentum/Read/ReadVariableOpReadVariableOp-SGD/module_wrapper_22/dense_1/kernel/momentum*
_output_shapes
:	ђ*
dtype0
«
+SGD/module_wrapper_22/dense_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+SGD/module_wrapper_22/dense_1/bias/momentum
Д
?SGD/module_wrapper_22/dense_1/bias/momentum/Read/ReadVariableOpReadVariableOp+SGD/module_wrapper_22/dense_1/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp
џ┼
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*н─
value╔─B┼─ Bй─
о
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer_with_weights-9
layer-20
layer-21
layer_with_weights-10
layer-22
layer-23
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
_
_module
 	variables
!regularization_losses
"trainable_variables
#	keras_api
_
$_module
%	variables
&regularization_losses
'trainable_variables
(	keras_api
_
)_module
*	variables
+regularization_losses
,trainable_variables
-	keras_api
_
._module
/	variables
0regularization_losses
1trainable_variables
2	keras_api
_
3_module
4	variables
5regularization_losses
6trainable_variables
7	keras_api
_
8_module
9	variables
:regularization_losses
;trainable_variables
<	keras_api
_
=_module
>	variables
?regularization_losses
@trainable_variables
A	keras_api
_
B_module
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
_
G_module
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
_
L_module
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
_
Q_module
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
_
V_module
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
_
[_module
\	variables
]regularization_losses
^trainable_variables
_	keras_api
_
`_module
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
_
e_module
f	variables
gregularization_losses
htrainable_variables
i	keras_api
_
j_module
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
_
o_module
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
_
t_module
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
_
y_module
z	variables
{regularization_losses
|trainable_variables
}	keras_api
b
~_module
	variables
ђregularization_losses
Ђtrainable_variables
ѓ	keras_api
d
Ѓ_module
ё	variables
Ёregularization_losses
єtrainable_variables
Є	keras_api
d
ѕ_module
Ѕ	variables
іregularization_losses
Іtrainable_variables
ї	keras_api
d
Ї_module
ј	variables
Јregularization_losses
љtrainable_variables
Љ	keras_api
d
њ_module
Њ	variables
ћregularization_losses
Ћtrainable_variables
ќ	keras_api
к
	Ќiter

ўdecay
Ўlearning_rate
џmomentumЏmomentumаюmomentumАЮmomentumбъmomentumБАmomentumцбmomentumЦБmomentumдцmomentumДДmomentumееmomentumЕЕmomentumффmomentumФГmomentumг«momentumГ»momentum«░momentum»│momentum░┤momentum▒хmomentum▓Хmomentum│╣momentum┤║momentumх
ќ
Џ0
ю1
Ю2
ъ3
Ъ4
а5
А6
б7
Б8
ц9
Ц10
д11
Д12
е13
Е14
ф15
Ф16
г17
Г18
«19
»20
░21
▒22
▓23
│24
┤25
х26
Х27
и28
И29
╣30
║31
 
╝
Џ0
ю1
Ю2
ъ3
А4
б5
Б6
ц7
Д8
е9
Е10
ф11
Г12
«13
»14
░15
│16
┤17
х18
Х19
╣20
║21
▓
	variables
regularization_losses
╗non_trainable_variables
trainable_variables
╝layers
йmetrics
Йlayer_metrics
 ┐layer_regularization_losses
 
n
Џkernel
	юbias
└	variables
┴regularization_losses
┬trainable_variables
├	keras_api

Џ0
ю1
 

Џ0
ю1
▓
 	variables
!regularization_losses
─non_trainable_variables
"trainable_variables
┼layers
кmetrics
Кlayer_metrics
 ╚layer_regularization_losses
V
╔	variables
╩regularization_losses
╦trainable_variables
╠	keras_api
 
 
 
▓
%	variables
&regularization_losses
═non_trainable_variables
'trainable_variables
╬layers
¤metrics
лlayer_metrics
 Лlayer_regularization_losses
а
	мaxis

Юgamma
	ъbeta
Ъmoving_mean
аmoving_variance
М	variables
нregularization_losses
Нtrainable_variables
о	keras_api
 
Ю0
ъ1
Ъ2
а3
 

Ю0
ъ1
▓
*	variables
+regularization_losses
Оnon_trainable_variables
,trainable_variables
пlayers
┘metrics
┌layer_metrics
 █layer_regularization_losses
V
▄	variables
Пregularization_losses
яtrainable_variables
▀	keras_api
 
 
 
▓
/	variables
0regularization_losses
Яnon_trainable_variables
1trainable_variables
рlayers
Рmetrics
сlayer_metrics
 Сlayer_regularization_losses
n
Аkernel
	бbias
т	variables
Тregularization_losses
уtrainable_variables
У	keras_api

А0
б1
 

А0
б1
▓
4	variables
5regularization_losses
жnon_trainable_variables
6trainable_variables
Жlayers
вmetrics
Вlayer_metrics
 ьlayer_regularization_losses
V
Ь	variables
№regularization_losses
­trainable_variables
ы	keras_api
 
 
 
▓
9	variables
:regularization_losses
Ыnon_trainable_variables
;trainable_variables
зlayers
Зmetrics
шlayer_metrics
 Шlayer_regularization_losses
а
	эaxis

Бgamma
	цbeta
Цmoving_mean
дmoving_variance
Э	variables
щregularization_losses
Щtrainable_variables
ч	keras_api
 
Б0
ц1
Ц2
д3
 

Б0
ц1
▓
>	variables
?regularization_losses
Чnon_trainable_variables
@trainable_variables
§layers
■metrics
 layer_metrics
 ђlayer_regularization_losses
V
Ђ	variables
ѓregularization_losses
Ѓtrainable_variables
ё	keras_api
 
 
 
▓
C	variables
Dregularization_losses
Ёnon_trainable_variables
Etrainable_variables
єlayers
Єmetrics
ѕlayer_metrics
 Ѕlayer_regularization_losses
n
Дkernel
	еbias
і	variables
Іregularization_losses
їtrainable_variables
Ї	keras_api

Д0
е1
 

Д0
е1
▓
H	variables
Iregularization_losses
јnon_trainable_variables
Jtrainable_variables
Јlayers
љmetrics
Љlayer_metrics
 њlayer_regularization_losses
V
Њ	variables
ћregularization_losses
Ћtrainable_variables
ќ	keras_api
 
 
 
▓
M	variables
Nregularization_losses
Ќnon_trainable_variables
Otrainable_variables
ўlayers
Ўmetrics
џlayer_metrics
 Џlayer_regularization_losses
а
	юaxis

Еgamma
	фbeta
Фmoving_mean
гmoving_variance
Ю	variables
ъregularization_losses
Ъtrainable_variables
а	keras_api
 
Е0
ф1
Ф2
г3
 

Е0
ф1
▓
R	variables
Sregularization_losses
Аnon_trainable_variables
Ttrainable_variables
бlayers
Бmetrics
цlayer_metrics
 Цlayer_regularization_losses
V
д	variables
Дregularization_losses
еtrainable_variables
Е	keras_api
 
 
 
▓
W	variables
Xregularization_losses
фnon_trainable_variables
Ytrainable_variables
Фlayers
гmetrics
Гlayer_metrics
 «layer_regularization_losses
n
Гkernel
	«bias
»	variables
░regularization_losses
▒trainable_variables
▓	keras_api

Г0
«1
 

Г0
«1
▓
\	variables
]regularization_losses
│non_trainable_variables
^trainable_variables
┤layers
хmetrics
Хlayer_metrics
 иlayer_regularization_losses
V
И	variables
╣regularization_losses
║trainable_variables
╗	keras_api
 
 
 
▓
a	variables
bregularization_losses
╝non_trainable_variables
ctrainable_variables
йlayers
Йmetrics
┐layer_metrics
 └layer_regularization_losses
а
	┴axis

»gamma
	░beta
▒moving_mean
▓moving_variance
┬	variables
├regularization_losses
─trainable_variables
┼	keras_api
 
»0
░1
▒2
▓3
 

»0
░1
▓
f	variables
gregularization_losses
кnon_trainable_variables
htrainable_variables
Кlayers
╚metrics
╔layer_metrics
 ╩layer_regularization_losses
V
╦	variables
╠regularization_losses
═trainable_variables
╬	keras_api
 
 
 
▓
k	variables
lregularization_losses
¤non_trainable_variables
mtrainable_variables
лlayers
Лmetrics
мlayer_metrics
 Мlayer_regularization_losses
V
н	variables
Нregularization_losses
оtrainable_variables
О	keras_api
 
 
 
▓
p	variables
qregularization_losses
пnon_trainable_variables
rtrainable_variables
┘layers
┌metrics
█layer_metrics
 ▄layer_regularization_losses
V
П	variables
яregularization_losses
▀trainable_variables
Я	keras_api
 
 
 
▓
u	variables
vregularization_losses
рnon_trainable_variables
wtrainable_variables
Рlayers
сmetrics
Сlayer_metrics
 тlayer_regularization_losses
n
│kernel
	┤bias
Т	variables
уregularization_losses
Уtrainable_variables
ж	keras_api

│0
┤1
 

│0
┤1
▓
z	variables
{regularization_losses
Жnon_trainable_variables
|trainable_variables
вlayers
Вmetrics
ьlayer_metrics
 Ьlayer_regularization_losses
V
№	variables
­regularization_losses
ыtrainable_variables
Ы	keras_api
 
 
 
┤
	variables
ђregularization_losses
зnon_trainable_variables
Ђtrainable_variables
Зlayers
шmetrics
Шlayer_metrics
 эlayer_regularization_losses
а
	Эaxis

хgamma
	Хbeta
иmoving_mean
Иmoving_variance
щ	variables
Щregularization_losses
чtrainable_variables
Ч	keras_api
 
х0
Х1
и2
И3
 

х0
Х1
х
ё	variables
Ёregularization_losses
§non_trainable_variables
єtrainable_variables
■layers
 metrics
ђlayer_metrics
 Ђlayer_regularization_losses
V
ѓ	variables
Ѓregularization_losses
ёtrainable_variables
Ё	keras_api
 
 
 
х
Ѕ	variables
іregularization_losses
єnon_trainable_variables
Іtrainable_variables
Єlayers
ѕmetrics
Ѕlayer_metrics
 іlayer_regularization_losses
n
╣kernel
	║bias
І	variables
їregularization_losses
Їtrainable_variables
ј	keras_api

╣0
║1
 

╣0
║1
х
ј	variables
Јregularization_losses
Јnon_trainable_variables
љtrainable_variables
љlayers
Љmetrics
њlayer_metrics
 Њlayer_regularization_losses
V
ћ	variables
Ћregularization_losses
ќtrainable_variables
Ќ	keras_api
 
 
 
х
Њ	variables
ћregularization_losses
ўnon_trainable_variables
Ћtrainable_variables
Ўlayers
џmetrics
Џlayer_metrics
 юlayer_regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEmodule_wrapper/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEmodule_wrapper/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*module_wrapper_2/batch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)module_wrapper_2/batch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0module_wrapper_2/batch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE4module_wrapper_2/batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE module_wrapper_4/conv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEmodule_wrapper_4/conv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE,module_wrapper_6/batch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+module_wrapper_6/batch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2module_wrapper_6/batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6module_wrapper_6/batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE module_wrapper_8/conv2d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEmodule_wrapper_8/conv2d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-module_wrapper_10/batch_normalization_2/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,module_wrapper_10/batch_normalization_2/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3module_wrapper_10/batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7module_wrapper_10/batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!module_wrapper_12/conv2d_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEmodule_wrapper_12/conv2d_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-module_wrapper_14/batch_normalization_3/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,module_wrapper_14/batch_normalization_3/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3module_wrapper_14/batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7module_wrapper_14/batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEmodule_wrapper_18/dense/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEmodule_wrapper_18/dense/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-module_wrapper_20/batch_normalization_4/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,module_wrapper_20/batch_normalization_4/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3module_wrapper_20/batch_normalization_4/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7module_wrapper_20/batch_normalization_4/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE module_wrapper_22/dense_1/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEmodule_wrapper_22/dense_1/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
P
Ъ0
а1
Ц2
д3
Ф4
г5
▒6
▓7
и8
И9
Х
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23

Ю0
ъ1
 
 

Џ0
ю1
 

Џ0
ю1
х
└	variables
┴regularization_losses
Ъnon_trainable_variables
┬trainable_variables
аlayers
Аmetrics
бlayer_metrics
 Бlayer_regularization_losses
 
 
 
 
 
 
 
 
х
╔	variables
╩regularization_losses
цnon_trainable_variables
╦trainable_variables
Цlayers
дmetrics
Дlayer_metrics
 еlayer_regularization_losses
 
 
 
 
 
 
 
Ю0
ъ1
Ъ2
а3
 

Ю0
ъ1
х
М	variables
нregularization_losses
Еnon_trainable_variables
Нtrainable_variables
фlayers
Фmetrics
гlayer_metrics
 Гlayer_regularization_losses

Ъ0
а1
 
 
 
 
 
 
 
х
▄	variables
Пregularization_losses
«non_trainable_variables
яtrainable_variables
»layers
░metrics
▒layer_metrics
 ▓layer_regularization_losses
 
 
 
 
 

А0
б1
 

А0
б1
х
т	variables
Тregularization_losses
│non_trainable_variables
уtrainable_variables
┤layers
хmetrics
Хlayer_metrics
 иlayer_regularization_losses
 
 
 
 
 
 
 
 
х
Ь	variables
№regularization_losses
Иnon_trainable_variables
­trainable_variables
╣layers
║metrics
╗layer_metrics
 ╝layer_regularization_losses
 
 
 
 
 
 
 
Б0
ц1
Ц2
д3
 

Б0
ц1
х
Э	variables
щregularization_losses
йnon_trainable_variables
Щtrainable_variables
Йlayers
┐metrics
└layer_metrics
 ┴layer_regularization_losses

Ц0
д1
 
 
 
 
 
 
 
х
Ђ	variables
ѓregularization_losses
┬non_trainable_variables
Ѓtrainable_variables
├layers
─metrics
┼layer_metrics
 кlayer_regularization_losses
 
 
 
 
 

Д0
е1
 

Д0
е1
х
і	variables
Іregularization_losses
Кnon_trainable_variables
їtrainable_variables
╚layers
╔metrics
╩layer_metrics
 ╦layer_regularization_losses
 
 
 
 
 
 
 
 
х
Њ	variables
ћregularization_losses
╠non_trainable_variables
Ћtrainable_variables
═layers
╬metrics
¤layer_metrics
 лlayer_regularization_losses
 
 
 
 
 
 
 
Е0
ф1
Ф2
г3
 

Е0
ф1
х
Ю	variables
ъregularization_losses
Лnon_trainable_variables
Ъtrainable_variables
мlayers
Мmetrics
нlayer_metrics
 Нlayer_regularization_losses

Ф0
г1
 
 
 
 
 
 
 
х
д	variables
Дregularization_losses
оnon_trainable_variables
еtrainable_variables
Оlayers
пmetrics
┘layer_metrics
 ┌layer_regularization_losses
 
 
 
 
 

Г0
«1
 

Г0
«1
х
»	variables
░regularization_losses
█non_trainable_variables
▒trainable_variables
▄layers
Пmetrics
яlayer_metrics
 ▀layer_regularization_losses
 
 
 
 
 
 
 
 
х
И	variables
╣regularization_losses
Яnon_trainable_variables
║trainable_variables
рlayers
Рmetrics
сlayer_metrics
 Сlayer_regularization_losses
 
 
 
 
 
 
 
»0
░1
▒2
▓3
 

»0
░1
х
┬	variables
├regularization_losses
тnon_trainable_variables
─trainable_variables
Тlayers
уmetrics
Уlayer_metrics
 жlayer_regularization_losses

▒0
▓1
 
 
 
 
 
 
 
х
╦	variables
╠regularization_losses
Жnon_trainable_variables
═trainable_variables
вlayers
Вmetrics
ьlayer_metrics
 Ьlayer_regularization_losses
 
 
 
 
 
 
 
 
х
н	variables
Нregularization_losses
№non_trainable_variables
оtrainable_variables
­layers
ыmetrics
Ыlayer_metrics
 зlayer_regularization_losses
 
 
 
 
 
 
 
 
х
П	variables
яregularization_losses
Зnon_trainable_variables
▀trainable_variables
шlayers
Шmetrics
эlayer_metrics
 Эlayer_regularization_losses
 
 
 
 
 

│0
┤1
 

│0
┤1
х
Т	variables
уregularization_losses
щnon_trainable_variables
Уtrainable_variables
Щlayers
чmetrics
Чlayer_metrics
 §layer_regularization_losses
 
 
 
 
 
 
 
 
х
№	variables
­regularization_losses
■non_trainable_variables
ыtrainable_variables
 layers
ђmetrics
Ђlayer_metrics
 ѓlayer_regularization_losses
 
 
 
 
 
 
 
х0
Х1
и2
И3
 

х0
Х1
х
щ	variables
Щregularization_losses
Ѓnon_trainable_variables
чtrainable_variables
ёlayers
Ёmetrics
єlayer_metrics
 Єlayer_regularization_losses

и0
И1
 
 
 
 
 
 
 
х
ѓ	variables
Ѓregularization_losses
ѕnon_trainable_variables
ёtrainable_variables
Ѕlayers
іmetrics
Іlayer_metrics
 їlayer_regularization_losses
 
 
 
 
 

╣0
║1
 

╣0
║1
х
І	variables
їregularization_losses
Їnon_trainable_variables
Їtrainable_variables
јlayers
Јmetrics
љlayer_metrics
 Љlayer_regularization_losses
 
 
 
 
 
 
 
 
х
ћ	variables
Ћregularization_losses
њnon_trainable_variables
ќtrainable_variables
Њlayers
ћmetrics
Ћlayer_metrics
 ќlayer_regularization_losses
 
 
 
 
 
8

Ќtotal

ўcount
Ў	variables
џ	keras_api
I

Џtotal

юcount
Ю
_fn_kwargs
ъ	variables
Ъ	keras_api
 
 
 
 
 
 
 
 
 
 

Ъ0
а1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Ц0
д1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Ф0
г1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

▒0
▓1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

и0
И1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ќ0
ў1

Ў	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Џ0
ю1

ъ	variables
Ѕє
VARIABLE_VALUE)SGD/module_wrapper/conv2d/kernel/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Єё
VARIABLE_VALUE'SGD/module_wrapper/conv2d/bias/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Ќћ
VARIABLE_VALUE7SGD/module_wrapper_2/batch_normalization/gamma/momentumIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ќЊ
VARIABLE_VALUE6SGD/module_wrapper_2/batch_normalization/beta/momentumIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Їі
VARIABLE_VALUE-SGD/module_wrapper_4/conv2d_1/kernel/momentumIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE+SGD/module_wrapper_4/conv2d_1/bias/momentumIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Ўќ
VARIABLE_VALUE9SGD/module_wrapper_6/batch_normalization_1/gamma/momentumIvariables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ўЋ
VARIABLE_VALUE8SGD/module_wrapper_6/batch_normalization_1/beta/momentumIvariables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUE-SGD/module_wrapper_8/conv2d_2/kernel/momentumJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE+SGD/module_wrapper_8/conv2d_2/bias/momentumJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Џў
VARIABLE_VALUE:SGD/module_wrapper_10/batch_normalization_2/gamma/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
џЌ
VARIABLE_VALUE9SGD/module_wrapper_10/batch_normalization_2/beta/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Јї
VARIABLE_VALUE.SGD/module_wrapper_12/conv2d_3/kernel/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Їі
VARIABLE_VALUE,SGD/module_wrapper_12/conv2d_3/bias/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Џў
VARIABLE_VALUE:SGD/module_wrapper_14/batch_normalization_3/gamma/momentumJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
џЌ
VARIABLE_VALUE9SGD/module_wrapper_14/batch_normalization_3/beta/momentumJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE+SGD/module_wrapper_18/dense/kernel/momentumJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE)SGD/module_wrapper_18/dense/bias/momentumJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Џў
VARIABLE_VALUE:SGD/module_wrapper_20/batch_normalization_4/gamma/momentumJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
џЌ
VARIABLE_VALUE9SGD/module_wrapper_20/batch_normalization_4/beta/momentumJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUE-SGD/module_wrapper_22/dense_1/kernel/momentumJvariables/30/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE+SGD/module_wrapper_22/dense_1/bias/momentumJvariables/31/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Ќ
$serving_default_module_wrapper_inputPlaceholder*/
_output_shapes
:         dd*
dtype0*$
shape:         dd
у
StatefulPartitionedCallStatefulPartitionedCall$serving_default_module_wrapper_inputmodule_wrapper/conv2d/kernelmodule_wrapper/conv2d/bias*module_wrapper_2/batch_normalization/gamma)module_wrapper_2/batch_normalization/beta0module_wrapper_2/batch_normalization/moving_mean4module_wrapper_2/batch_normalization/moving_variance module_wrapper_4/conv2d_1/kernelmodule_wrapper_4/conv2d_1/bias,module_wrapper_6/batch_normalization_1/gamma+module_wrapper_6/batch_normalization_1/beta2module_wrapper_6/batch_normalization_1/moving_mean6module_wrapper_6/batch_normalization_1/moving_variance module_wrapper_8/conv2d_2/kernelmodule_wrapper_8/conv2d_2/bias-module_wrapper_10/batch_normalization_2/gamma,module_wrapper_10/batch_normalization_2/beta3module_wrapper_10/batch_normalization_2/moving_mean7module_wrapper_10/batch_normalization_2/moving_variance!module_wrapper_12/conv2d_3/kernelmodule_wrapper_12/conv2d_3/bias-module_wrapper_14/batch_normalization_3/gamma,module_wrapper_14/batch_normalization_3/beta3module_wrapper_14/batch_normalization_3/moving_mean7module_wrapper_14/batch_normalization_3/moving_variancemodule_wrapper_18/dense/kernelmodule_wrapper_18/dense/bias3module_wrapper_20/batch_normalization_4/moving_mean7module_wrapper_20/batch_normalization_4/moving_variance,module_wrapper_20/batch_normalization_4/beta-module_wrapper_20/batch_normalization_4/gamma module_wrapper_22/dense_1/kernelmodule_wrapper_22/dense_1/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference_signature_wrapper_58232
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
п 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp0module_wrapper/conv2d/kernel/Read/ReadVariableOp.module_wrapper/conv2d/bias/Read/ReadVariableOp>module_wrapper_2/batch_normalization/gamma/Read/ReadVariableOp=module_wrapper_2/batch_normalization/beta/Read/ReadVariableOpDmodule_wrapper_2/batch_normalization/moving_mean/Read/ReadVariableOpHmodule_wrapper_2/batch_normalization/moving_variance/Read/ReadVariableOp4module_wrapper_4/conv2d_1/kernel/Read/ReadVariableOp2module_wrapper_4/conv2d_1/bias/Read/ReadVariableOp@module_wrapper_6/batch_normalization_1/gamma/Read/ReadVariableOp?module_wrapper_6/batch_normalization_1/beta/Read/ReadVariableOpFmodule_wrapper_6/batch_normalization_1/moving_mean/Read/ReadVariableOpJmodule_wrapper_6/batch_normalization_1/moving_variance/Read/ReadVariableOp4module_wrapper_8/conv2d_2/kernel/Read/ReadVariableOp2module_wrapper_8/conv2d_2/bias/Read/ReadVariableOpAmodule_wrapper_10/batch_normalization_2/gamma/Read/ReadVariableOp@module_wrapper_10/batch_normalization_2/beta/Read/ReadVariableOpGmodule_wrapper_10/batch_normalization_2/moving_mean/Read/ReadVariableOpKmodule_wrapper_10/batch_normalization_2/moving_variance/Read/ReadVariableOp5module_wrapper_12/conv2d_3/kernel/Read/ReadVariableOp3module_wrapper_12/conv2d_3/bias/Read/ReadVariableOpAmodule_wrapper_14/batch_normalization_3/gamma/Read/ReadVariableOp@module_wrapper_14/batch_normalization_3/beta/Read/ReadVariableOpGmodule_wrapper_14/batch_normalization_3/moving_mean/Read/ReadVariableOpKmodule_wrapper_14/batch_normalization_3/moving_variance/Read/ReadVariableOp2module_wrapper_18/dense/kernel/Read/ReadVariableOp0module_wrapper_18/dense/bias/Read/ReadVariableOpAmodule_wrapper_20/batch_normalization_4/gamma/Read/ReadVariableOp@module_wrapper_20/batch_normalization_4/beta/Read/ReadVariableOpGmodule_wrapper_20/batch_normalization_4/moving_mean/Read/ReadVariableOpKmodule_wrapper_20/batch_normalization_4/moving_variance/Read/ReadVariableOp4module_wrapper_22/dense_1/kernel/Read/ReadVariableOp2module_wrapper_22/dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp=SGD/module_wrapper/conv2d/kernel/momentum/Read/ReadVariableOp;SGD/module_wrapper/conv2d/bias/momentum/Read/ReadVariableOpKSGD/module_wrapper_2/batch_normalization/gamma/momentum/Read/ReadVariableOpJSGD/module_wrapper_2/batch_normalization/beta/momentum/Read/ReadVariableOpASGD/module_wrapper_4/conv2d_1/kernel/momentum/Read/ReadVariableOp?SGD/module_wrapper_4/conv2d_1/bias/momentum/Read/ReadVariableOpMSGD/module_wrapper_6/batch_normalization_1/gamma/momentum/Read/ReadVariableOpLSGD/module_wrapper_6/batch_normalization_1/beta/momentum/Read/ReadVariableOpASGD/module_wrapper_8/conv2d_2/kernel/momentum/Read/ReadVariableOp?SGD/module_wrapper_8/conv2d_2/bias/momentum/Read/ReadVariableOpNSGD/module_wrapper_10/batch_normalization_2/gamma/momentum/Read/ReadVariableOpMSGD/module_wrapper_10/batch_normalization_2/beta/momentum/Read/ReadVariableOpBSGD/module_wrapper_12/conv2d_3/kernel/momentum/Read/ReadVariableOp@SGD/module_wrapper_12/conv2d_3/bias/momentum/Read/ReadVariableOpNSGD/module_wrapper_14/batch_normalization_3/gamma/momentum/Read/ReadVariableOpMSGD/module_wrapper_14/batch_normalization_3/beta/momentum/Read/ReadVariableOp?SGD/module_wrapper_18/dense/kernel/momentum/Read/ReadVariableOp=SGD/module_wrapper_18/dense/bias/momentum/Read/ReadVariableOpNSGD/module_wrapper_20/batch_normalization_4/gamma/momentum/Read/ReadVariableOpMSGD/module_wrapper_20/batch_normalization_4/beta/momentum/Read/ReadVariableOpASGD/module_wrapper_22/dense_1/kernel/momentum/Read/ReadVariableOp?SGD/module_wrapper_22/dense_1/bias/momentum/Read/ReadVariableOpConst*K
TinD
B2@	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *'
f"R 
__inference__traced_save_60734
ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameSGD/iter	SGD/decaySGD/learning_rateSGD/momentummodule_wrapper/conv2d/kernelmodule_wrapper/conv2d/bias*module_wrapper_2/batch_normalization/gamma)module_wrapper_2/batch_normalization/beta0module_wrapper_2/batch_normalization/moving_mean4module_wrapper_2/batch_normalization/moving_variance module_wrapper_4/conv2d_1/kernelmodule_wrapper_4/conv2d_1/bias,module_wrapper_6/batch_normalization_1/gamma+module_wrapper_6/batch_normalization_1/beta2module_wrapper_6/batch_normalization_1/moving_mean6module_wrapper_6/batch_normalization_1/moving_variance module_wrapper_8/conv2d_2/kernelmodule_wrapper_8/conv2d_2/bias-module_wrapper_10/batch_normalization_2/gamma,module_wrapper_10/batch_normalization_2/beta3module_wrapper_10/batch_normalization_2/moving_mean7module_wrapper_10/batch_normalization_2/moving_variance!module_wrapper_12/conv2d_3/kernelmodule_wrapper_12/conv2d_3/bias-module_wrapper_14/batch_normalization_3/gamma,module_wrapper_14/batch_normalization_3/beta3module_wrapper_14/batch_normalization_3/moving_mean7module_wrapper_14/batch_normalization_3/moving_variancemodule_wrapper_18/dense/kernelmodule_wrapper_18/dense/bias-module_wrapper_20/batch_normalization_4/gamma,module_wrapper_20/batch_normalization_4/beta3module_wrapper_20/batch_normalization_4/moving_mean7module_wrapper_20/batch_normalization_4/moving_variance module_wrapper_22/dense_1/kernelmodule_wrapper_22/dense_1/biastotalcounttotal_1count_1)SGD/module_wrapper/conv2d/kernel/momentum'SGD/module_wrapper/conv2d/bias/momentum7SGD/module_wrapper_2/batch_normalization/gamma/momentum6SGD/module_wrapper_2/batch_normalization/beta/momentum-SGD/module_wrapper_4/conv2d_1/kernel/momentum+SGD/module_wrapper_4/conv2d_1/bias/momentum9SGD/module_wrapper_6/batch_normalization_1/gamma/momentum8SGD/module_wrapper_6/batch_normalization_1/beta/momentum-SGD/module_wrapper_8/conv2d_2/kernel/momentum+SGD/module_wrapper_8/conv2d_2/bias/momentum:SGD/module_wrapper_10/batch_normalization_2/gamma/momentum9SGD/module_wrapper_10/batch_normalization_2/beta/momentum.SGD/module_wrapper_12/conv2d_3/kernel/momentum,SGD/module_wrapper_12/conv2d_3/bias/momentum:SGD/module_wrapper_14/batch_normalization_3/gamma/momentum9SGD/module_wrapper_14/batch_normalization_3/beta/momentum+SGD/module_wrapper_18/dense/kernel/momentum)SGD/module_wrapper_18/dense/bias/momentum:SGD/module_wrapper_20/batch_normalization_4/gamma/momentum9SGD/module_wrapper_20/batch_normalization_4/beta/momentum-SGD/module_wrapper_22/dense_1/kernel/momentum+SGD/module_wrapper_22/dense_1/bias/momentum*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__traced_restore_60930ъЫ
кС
Й$
E__inference_sequential_layer_call_and_return_conditional_losses_59073

inputsN
4module_wrapper_conv2d_conv2d_readvariableop_resource: C
5module_wrapper_conv2d_biasadd_readvariableop_resource: J
<module_wrapper_2_batch_normalization_readvariableop_resource: L
>module_wrapper_2_batch_normalization_readvariableop_1_resource: [
Mmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resource: ]
Omodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: R
8module_wrapper_4_conv2d_1_conv2d_readvariableop_resource: @G
9module_wrapper_4_conv2d_1_biasadd_readvariableop_resource:@L
>module_wrapper_6_batch_normalization_1_readvariableop_resource:@N
@module_wrapper_6_batch_normalization_1_readvariableop_1_resource:@]
Omodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@_
Qmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@R
8module_wrapper_8_conv2d_2_conv2d_readvariableop_resource:@@G
9module_wrapper_8_conv2d_2_biasadd_readvariableop_resource:@M
?module_wrapper_10_batch_normalization_2_readvariableop_resource:@O
Amodule_wrapper_10_batch_normalization_2_readvariableop_1_resource:@^
Pmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@`
Rmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@T
9module_wrapper_12_conv2d_3_conv2d_readvariableop_resource:@ђI
:module_wrapper_12_conv2d_3_biasadd_readvariableop_resource:	ђN
?module_wrapper_14_batch_normalization_3_readvariableop_resource:	ђP
Amodule_wrapper_14_batch_normalization_3_readvariableop_1_resource:	ђ_
Pmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђa
Rmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђJ
6module_wrapper_18_dense_matmul_readvariableop_resource:
ђђF
7module_wrapper_18_dense_biasadd_readvariableop_resource:	ђS
Dmodule_wrapper_20_batch_normalization_4_cast_readvariableop_resource:	ђU
Fmodule_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource:	ђU
Fmodule_wrapper_20_batch_normalization_4_cast_2_readvariableop_resource:	ђU
Fmodule_wrapper_20_batch_normalization_4_cast_3_readvariableop_resource:	ђK
8module_wrapper_22_dense_1_matmul_readvariableop_resource:	ђG
9module_wrapper_22_dense_1_biasadd_readvariableop_resource:
identityѕб,module_wrapper/conv2d/BiasAdd/ReadVariableOpб+module_wrapper/conv2d/Conv2D/ReadVariableOpбGmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбImodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б6module_wrapper_10/batch_normalization_2/ReadVariableOpб8module_wrapper_10/batch_normalization_2/ReadVariableOp_1б1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpб0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpбGmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбImodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б6module_wrapper_14/batch_normalization_3/ReadVariableOpб8module_wrapper_14/batch_normalization_3/ReadVariableOp_1б.module_wrapper_18/dense/BiasAdd/ReadVariableOpб-module_wrapper_18/dense/MatMul/ReadVariableOpбDmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpбFmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1б3module_wrapper_2/batch_normalization/ReadVariableOpб5module_wrapper_2/batch_normalization/ReadVariableOp_1б;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpб=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpб=module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpб=module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpб0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpб/module_wrapper_22/dense_1/MatMul/ReadVariableOpб0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpб/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpбFmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбHmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б5module_wrapper_6/batch_normalization_1/ReadVariableOpб7module_wrapper_6/batch_normalization_1/ReadVariableOp_1б0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpб/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpО
+module_wrapper/conv2d/Conv2D/ReadVariableOpReadVariableOp4module_wrapper_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+module_wrapper/conv2d/Conv2D/ReadVariableOpт
module_wrapper/conv2d/Conv2DConv2Dinputs3module_wrapper/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2
module_wrapper/conv2d/Conv2D╬
,module_wrapper/conv2d/BiasAdd/ReadVariableOpReadVariableOp5module_wrapper_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,module_wrapper/conv2d/BiasAdd/ReadVariableOpЯ
module_wrapper/conv2d/BiasAddBiasAdd%module_wrapper/conv2d/Conv2D:output:04module_wrapper/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd 2
module_wrapper/conv2d/BiasAdd«
 module_wrapper_1/activation/ReluRelu&module_wrapper/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         dd 2"
 module_wrapper_1/activation/Reluс
3module_wrapper_2/batch_normalization/ReadVariableOpReadVariableOp<module_wrapper_2_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype025
3module_wrapper_2/batch_normalization/ReadVariableOpж
5module_wrapper_2/batch_normalization/ReadVariableOp_1ReadVariableOp>module_wrapper_2_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype027
5module_wrapper_2/batch_normalization/ReadVariableOp_1ќ
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpю
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1л
5module_wrapper_2/batch_normalization/FusedBatchNormV3FusedBatchNormV3.module_wrapper_1/activation/Relu:activations:0;module_wrapper_2/batch_normalization/ReadVariableOp:value:0=module_wrapper_2/batch_normalization/ReadVariableOp_1:value:0Lmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Nmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oЃ:*
is_training( 27
5module_wrapper_2/batch_normalization/FusedBatchNormV3Ѓ
&module_wrapper_3/max_pooling2d/MaxPoolMaxPool9module_wrapper_2/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2(
&module_wrapper_3/max_pooling2d/MaxPoolс
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_4_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype021
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpџ
 module_wrapper_4/conv2d_1/Conv2DConv2D/module_wrapper_3/max_pooling2d/MaxPool:output:07module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2"
 module_wrapper_4/conv2d_1/Conv2D┌
0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_4_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp­
!module_wrapper_4/conv2d_1/BiasAddBiasAdd)module_wrapper_4/conv2d_1/Conv2D:output:08module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2#
!module_wrapper_4/conv2d_1/BiasAddХ
"module_wrapper_5/activation_1/ReluRelu*module_wrapper_4/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         !!@2$
"module_wrapper_5/activation_1/Reluж
5module_wrapper_6/batch_normalization_1/ReadVariableOpReadVariableOp>module_wrapper_6_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype027
5module_wrapper_6/batch_normalization_1/ReadVariableOp№
7module_wrapper_6/batch_normalization_1/ReadVariableOp_1ReadVariableOp@module_wrapper_6_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7module_wrapper_6/batch_normalization_1/ReadVariableOp_1ю
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02H
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpб
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1я
7module_wrapper_6/batch_normalization_1/FusedBatchNormV3FusedBatchNormV30module_wrapper_5/activation_1/Relu:activations:0=module_wrapper_6/batch_normalization_1/ReadVariableOp:value:0?module_wrapper_6/batch_normalization_1/ReadVariableOp_1:value:0Nmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Pmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !!@:@:@:@:@:*
epsilon%oЃ:*
is_training( 29
7module_wrapper_6/batch_normalization_1/FusedBatchNormV3Ѕ
(module_wrapper_7/max_pooling2d_1/MaxPoolMaxPool;module_wrapper_6/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2*
(module_wrapper_7/max_pooling2d_1/MaxPoolс
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_8_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype021
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpю
 module_wrapper_8/conv2d_2/Conv2DConv2D1module_wrapper_7/max_pooling2d_1/MaxPool:output:07module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2"
 module_wrapper_8/conv2d_2/Conv2D┌
0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_8_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp­
!module_wrapper_8/conv2d_2/BiasAddBiasAdd)module_wrapper_8/conv2d_2/Conv2D:output:08module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2#
!module_wrapper_8/conv2d_2/BiasAddХ
"module_wrapper_9/activation_2/ReluRelu*module_wrapper_8/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2$
"module_wrapper_9/activation_2/ReluВ
6module_wrapper_10/batch_normalization_2/ReadVariableOpReadVariableOp?module_wrapper_10_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype028
6module_wrapper_10/batch_normalization_2/ReadVariableOpЫ
8module_wrapper_10/batch_normalization_2/ReadVariableOp_1ReadVariableOpAmodule_wrapper_10_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8module_wrapper_10/batch_normalization_2/ReadVariableOp_1Ъ
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1С
8module_wrapper_10/batch_normalization_2/FusedBatchNormV3FusedBatchNormV30module_wrapper_9/activation_2/Relu:activations:0>module_wrapper_10/batch_normalization_2/ReadVariableOp:value:0@module_wrapper_10/batch_normalization_2/ReadVariableOp_1:value:0Omodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Qmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2:
8module_wrapper_10/batch_normalization_2/FusedBatchNormV3ї
)module_wrapper_11/max_pooling2d_2/MaxPoolMaxPool<module_wrapper_10/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2+
)module_wrapper_11/max_pooling2d_2/MaxPoolу
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpReadVariableOp9module_wrapper_12_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype022
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpА
!module_wrapper_12/conv2d_3/Conv2DConv2D2module_wrapper_11/max_pooling2d_2/MaxPool:output:08module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2#
!module_wrapper_12/conv2d_3/Conv2Dя
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp:module_wrapper_12_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpш
"module_wrapper_12/conv2d_3/BiasAddBiasAdd*module_wrapper_12/conv2d_3/Conv2D:output:09module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2$
"module_wrapper_12/conv2d_3/BiasAdd║
#module_wrapper_13/activation_3/ReluRelu+module_wrapper_12/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2%
#module_wrapper_13/activation_3/Reluь
6module_wrapper_14/batch_normalization_3/ReadVariableOpReadVariableOp?module_wrapper_14_batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6module_wrapper_14/batch_normalization_3/ReadVariableOpз
8module_wrapper_14/batch_normalization_3/ReadVariableOp_1ReadVariableOpAmodule_wrapper_14_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02:
8module_wrapper_14/batch_normalization_3/ReadVariableOp_1а
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02I
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpд
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02K
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ж
8module_wrapper_14/batch_normalization_3/FusedBatchNormV3FusedBatchNormV31module_wrapper_13/activation_3/Relu:activations:0>module_wrapper_14/batch_normalization_3/ReadVariableOp:value:0@module_wrapper_14/batch_normalization_3/ReadVariableOp_1:value:0Omodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Qmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2:
8module_wrapper_14/batch_normalization_3/FusedBatchNormV3Ї
)module_wrapper_15/max_pooling2d_3/MaxPoolMaxPool<module_wrapper_14/batch_normalization_3/FusedBatchNormV3:y:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2+
)module_wrapper_15/max_pooling2d_3/MaxPool├
"module_wrapper_16/dropout/IdentityIdentity2module_wrapper_15/max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2$
"module_wrapper_16/dropout/IdentityЊ
module_wrapper_17/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
module_wrapper_17/flatten/Const█
!module_wrapper_17/flatten/ReshapeReshape+module_wrapper_16/dropout/Identity:output:0(module_wrapper_17/flatten/Const:output:0*
T0*(
_output_shapes
:         ђ2#
!module_wrapper_17/flatten/ReshapeО
-module_wrapper_18/dense/MatMul/ReadVariableOpReadVariableOp6module_wrapper_18_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02/
-module_wrapper_18/dense/MatMul/ReadVariableOpЯ
module_wrapper_18/dense/MatMulMatMul*module_wrapper_17/flatten/Reshape:output:05module_wrapper_18/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2 
module_wrapper_18/dense/MatMulН
.module_wrapper_18/dense/BiasAdd/ReadVariableOpReadVariableOp7module_wrapper_18_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.module_wrapper_18/dense/BiasAdd/ReadVariableOpР
module_wrapper_18/dense/BiasAddBiasAdd(module_wrapper_18/dense/MatMul:product:06module_wrapper_18/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2!
module_wrapper_18/dense/BiasAdd»
#module_wrapper_19/activation_4/ReluRelu(module_wrapper_18/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2%
#module_wrapper_19/activation_4/ReluЧ
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpReadVariableOpDmodule_wrapper_20_batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02=
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpѓ
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpReadVariableOpFmodule_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpѓ
=module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpReadVariableOpFmodule_wrapper_20_batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpѓ
=module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpReadVariableOpFmodule_wrapper_20_batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpи
7module_wrapper_20/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:29
7module_wrapper_20/batch_normalization_4/batchnorm/add/yд
5module_wrapper_20/batch_normalization_4/batchnorm/addAddV2Emodule_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp:value:0@module_wrapper_20/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ27
5module_wrapper_20/batch_normalization_4/batchnorm/add▄
7module_wrapper_20/batch_normalization_4/batchnorm/RsqrtRsqrt9module_wrapper_20/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ29
7module_wrapper_20/batch_normalization_4/batchnorm/RsqrtЪ
5module_wrapper_20/batch_normalization_4/batchnorm/mulMul;module_wrapper_20/batch_normalization_4/batchnorm/Rsqrt:y:0Emodule_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ27
5module_wrapper_20/batch_normalization_4/batchnorm/mulџ
7module_wrapper_20/batch_normalization_4/batchnorm/mul_1Mul1module_wrapper_19/activation_4/Relu:activations:09module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ29
7module_wrapper_20/batch_normalization_4/batchnorm/mul_1Ъ
7module_wrapper_20/batch_normalization_4/batchnorm/mul_2MulCmodule_wrapper_20/batch_normalization_4/Cast/ReadVariableOp:value:09module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ29
7module_wrapper_20/batch_normalization_4/batchnorm/mul_2Ъ
5module_wrapper_20/batch_normalization_4/batchnorm/subSubEmodule_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOp:value:0;module_wrapper_20/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ27
5module_wrapper_20/batch_normalization_4/batchnorm/subд
7module_wrapper_20/batch_normalization_4/batchnorm/add_1AddV2;module_wrapper_20/batch_normalization_4/batchnorm/mul_1:z:09module_wrapper_20/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ29
7module_wrapper_20/batch_normalization_4/batchnorm/add_1╚
$module_wrapper_21/dropout_1/IdentityIdentity;module_wrapper_20/batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         ђ2&
$module_wrapper_21/dropout_1/Identity▄
/module_wrapper_22/dense_1/MatMul/ReadVariableOpReadVariableOp8module_wrapper_22_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype021
/module_wrapper_22/dense_1/MatMul/ReadVariableOpУ
 module_wrapper_22/dense_1/MatMulMatMul-module_wrapper_21/dropout_1/Identity:output:07module_wrapper_22/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 module_wrapper_22/dense_1/MatMul┌
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_22_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpж
!module_wrapper_22/dense_1/BiasAddBiasAdd*module_wrapper_22/dense_1/MatMul:product:08module_wrapper_22/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2#
!module_wrapper_22/dense_1/BiasAdd╣
&module_wrapper_23/activation_5/SoftmaxSoftmax*module_wrapper_22/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2(
&module_wrapper_23/activation_5/SoftmaxЬ
IdentityIdentity0module_wrapper_23/activation_5/Softmax:softmax:0-^module_wrapper/conv2d/BiasAdd/ReadVariableOp,^module_wrapper/conv2d/Conv2D/ReadVariableOpH^module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpJ^module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17^module_wrapper_10/batch_normalization_2/ReadVariableOp9^module_wrapper_10/batch_normalization_2/ReadVariableOp_12^module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp1^module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpH^module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpJ^module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17^module_wrapper_14/batch_normalization_3/ReadVariableOp9^module_wrapper_14/batch_normalization_3/ReadVariableOp_1/^module_wrapper_18/dense/BiasAdd/ReadVariableOp.^module_wrapper_18/dense/MatMul/ReadVariableOpE^module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpG^module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_14^module_wrapper_2/batch_normalization/ReadVariableOp6^module_wrapper_2/batch_normalization/ReadVariableOp_1<^module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp>^module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp>^module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOp>^module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp1^module_wrapper_22/dense_1/BiasAdd/ReadVariableOp0^module_wrapper_22/dense_1/MatMul/ReadVariableOp1^module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp0^module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpG^module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpI^module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_16^module_wrapper_6/batch_normalization_1/ReadVariableOp8^module_wrapper_6/batch_normalization_1/ReadVariableOp_11^module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp0^module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,module_wrapper/conv2d/BiasAdd/ReadVariableOp,module_wrapper/conv2d/BiasAdd/ReadVariableOp2Z
+module_wrapper/conv2d/Conv2D/ReadVariableOp+module_wrapper/conv2d/Conv2D/ReadVariableOp2њ
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpGmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2ќ
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12p
6module_wrapper_10/batch_normalization_2/ReadVariableOp6module_wrapper_10/batch_normalization_2/ReadVariableOp2t
8module_wrapper_10/batch_normalization_2/ReadVariableOp_18module_wrapper_10/batch_normalization_2/ReadVariableOp_12f
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp2d
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp2њ
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpGmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2ќ
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12p
6module_wrapper_14/batch_normalization_3/ReadVariableOp6module_wrapper_14/batch_normalization_3/ReadVariableOp2t
8module_wrapper_14/batch_normalization_3/ReadVariableOp_18module_wrapper_14/batch_normalization_3/ReadVariableOp_12`
.module_wrapper_18/dense/BiasAdd/ReadVariableOp.module_wrapper_18/dense/BiasAdd/ReadVariableOp2^
-module_wrapper_18/dense/MatMul/ReadVariableOp-module_wrapper_18/dense/MatMul/ReadVariableOp2ї
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpDmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp2љ
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3module_wrapper_2/batch_normalization/ReadVariableOp3module_wrapper_2/batch_normalization/ReadVariableOp2n
5module_wrapper_2/batch_normalization/ReadVariableOp_15module_wrapper_2/batch_normalization/ReadVariableOp_12z
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp2~
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp2~
=module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOp=module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOp2~
=module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp=module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp2d
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOp0module_wrapper_22/dense_1/BiasAdd/ReadVariableOp2b
/module_wrapper_22/dense_1/MatMul/ReadVariableOp/module_wrapper_22/dense_1/MatMul/ReadVariableOp2d
0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp2b
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp2љ
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpFmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2ћ
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5module_wrapper_6/batch_normalization_1/ReadVariableOp5module_wrapper_6/batch_normalization_1/ReadVariableOp2r
7module_wrapper_6/batch_normalization_1/ReadVariableOp_17module_wrapper_6/batch_normalization_1/ReadVariableOp_12d
0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp2b
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Т
Є
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_60465

inputs+
cast_readvariableop_resource:	ђ-
cast_1_readvariableop_resource:	ђ-
cast_2_readvariableop_resource:	ђ-
cast_3_readvariableop_resource:	ђ
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
з
h
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_57263

args_0
identity│
max_pooling2d_3/MaxPoolMaxPoolargs_0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool}
IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
З

ў
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_59984

args_08
$dense_matmul_readvariableop_resource:
ђђ4
%dense_biasadd_readvariableop_resource:	ђ
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense/MatMul/ReadVariableOpє
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/BiasAddе
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ж
L
0__inference_module_wrapper_9_layer_call_fn_59703

args_0
identityЛ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_574302
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
«╬
ц)
E__inference_sequential_layer_call_and_return_conditional_losses_59227

inputsN
4module_wrapper_conv2d_conv2d_readvariableop_resource: C
5module_wrapper_conv2d_biasadd_readvariableop_resource: J
<module_wrapper_2_batch_normalization_readvariableop_resource: L
>module_wrapper_2_batch_normalization_readvariableop_1_resource: [
Mmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resource: ]
Omodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: R
8module_wrapper_4_conv2d_1_conv2d_readvariableop_resource: @G
9module_wrapper_4_conv2d_1_biasadd_readvariableop_resource:@L
>module_wrapper_6_batch_normalization_1_readvariableop_resource:@N
@module_wrapper_6_batch_normalization_1_readvariableop_1_resource:@]
Omodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@_
Qmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@R
8module_wrapper_8_conv2d_2_conv2d_readvariableop_resource:@@G
9module_wrapper_8_conv2d_2_biasadd_readvariableop_resource:@M
?module_wrapper_10_batch_normalization_2_readvariableop_resource:@O
Amodule_wrapper_10_batch_normalization_2_readvariableop_1_resource:@^
Pmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@`
Rmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@T
9module_wrapper_12_conv2d_3_conv2d_readvariableop_resource:@ђI
:module_wrapper_12_conv2d_3_biasadd_readvariableop_resource:	ђN
?module_wrapper_14_batch_normalization_3_readvariableop_resource:	ђP
Amodule_wrapper_14_batch_normalization_3_readvariableop_1_resource:	ђ_
Pmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђa
Rmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђJ
6module_wrapper_18_dense_matmul_readvariableop_resource:
ђђF
7module_wrapper_18_dense_biasadd_readvariableop_resource:	ђ^
Omodule_wrapper_20_batch_normalization_4_assignmovingavg_readvariableop_resource:	ђ`
Qmodule_wrapper_20_batch_normalization_4_assignmovingavg_1_readvariableop_resource:	ђS
Dmodule_wrapper_20_batch_normalization_4_cast_readvariableop_resource:	ђU
Fmodule_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource:	ђK
8module_wrapper_22_dense_1_matmul_readvariableop_resource:	ђG
9module_wrapper_22_dense_1_biasadd_readvariableop_resource:
identityѕб,module_wrapper/conv2d/BiasAdd/ReadVariableOpб+module_wrapper/conv2d/Conv2D/ReadVariableOpб6module_wrapper_10/batch_normalization_2/AssignNewValueб8module_wrapper_10/batch_normalization_2/AssignNewValue_1бGmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбImodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б6module_wrapper_10/batch_normalization_2/ReadVariableOpб8module_wrapper_10/batch_normalization_2/ReadVariableOp_1б1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpб0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpб6module_wrapper_14/batch_normalization_3/AssignNewValueб8module_wrapper_14/batch_normalization_3/AssignNewValue_1бGmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбImodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б6module_wrapper_14/batch_normalization_3/ReadVariableOpб8module_wrapper_14/batch_normalization_3/ReadVariableOp_1б.module_wrapper_18/dense/BiasAdd/ReadVariableOpб-module_wrapper_18/dense/MatMul/ReadVariableOpб3module_wrapper_2/batch_normalization/AssignNewValueб5module_wrapper_2/batch_normalization/AssignNewValue_1бDmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpбFmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1б3module_wrapper_2/batch_normalization/ReadVariableOpб5module_wrapper_2/batch_normalization/ReadVariableOp_1б7module_wrapper_20/batch_normalization_4/AssignMovingAvgбFmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOpб9module_wrapper_20/batch_normalization_4/AssignMovingAvg_1бHmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpб;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpб=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpб0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpб/module_wrapper_22/dense_1/MatMul/ReadVariableOpб0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpб/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpб5module_wrapper_6/batch_normalization_1/AssignNewValueб7module_wrapper_6/batch_normalization_1/AssignNewValue_1бFmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбHmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б5module_wrapper_6/batch_normalization_1/ReadVariableOpб7module_wrapper_6/batch_normalization_1/ReadVariableOp_1б0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpб/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpО
+module_wrapper/conv2d/Conv2D/ReadVariableOpReadVariableOp4module_wrapper_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+module_wrapper/conv2d/Conv2D/ReadVariableOpт
module_wrapper/conv2d/Conv2DConv2Dinputs3module_wrapper/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2
module_wrapper/conv2d/Conv2D╬
,module_wrapper/conv2d/BiasAdd/ReadVariableOpReadVariableOp5module_wrapper_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,module_wrapper/conv2d/BiasAdd/ReadVariableOpЯ
module_wrapper/conv2d/BiasAddBiasAdd%module_wrapper/conv2d/Conv2D:output:04module_wrapper/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd 2
module_wrapper/conv2d/BiasAdd«
 module_wrapper_1/activation/ReluRelu&module_wrapper/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         dd 2"
 module_wrapper_1/activation/Reluс
3module_wrapper_2/batch_normalization/ReadVariableOpReadVariableOp<module_wrapper_2_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype025
3module_wrapper_2/batch_normalization/ReadVariableOpж
5module_wrapper_2/batch_normalization/ReadVariableOp_1ReadVariableOp>module_wrapper_2_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype027
5module_wrapper_2/batch_normalization/ReadVariableOp_1ќ
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpю
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1я
5module_wrapper_2/batch_normalization/FusedBatchNormV3FusedBatchNormV3.module_wrapper_1/activation/Relu:activations:0;module_wrapper_2/batch_normalization/ReadVariableOp:value:0=module_wrapper_2/batch_normalization/ReadVariableOp_1:value:0Lmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Nmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<27
5module_wrapper_2/batch_normalization/FusedBatchNormV3ч
3module_wrapper_2/batch_normalization/AssignNewValueAssignVariableOpMmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resourceBmodule_wrapper_2/batch_normalization/FusedBatchNormV3:batch_mean:0E^module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3module_wrapper_2/batch_normalization/AssignNewValueЄ
5module_wrapper_2/batch_normalization/AssignNewValue_1AssignVariableOpOmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceFmodule_wrapper_2/batch_normalization/FusedBatchNormV3:batch_variance:0G^module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5module_wrapper_2/batch_normalization/AssignNewValue_1Ѓ
&module_wrapper_3/max_pooling2d/MaxPoolMaxPool9module_wrapper_2/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2(
&module_wrapper_3/max_pooling2d/MaxPoolс
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_4_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype021
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpџ
 module_wrapper_4/conv2d_1/Conv2DConv2D/module_wrapper_3/max_pooling2d/MaxPool:output:07module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2"
 module_wrapper_4/conv2d_1/Conv2D┌
0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_4_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp­
!module_wrapper_4/conv2d_1/BiasAddBiasAdd)module_wrapper_4/conv2d_1/Conv2D:output:08module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2#
!module_wrapper_4/conv2d_1/BiasAddХ
"module_wrapper_5/activation_1/ReluRelu*module_wrapper_4/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         !!@2$
"module_wrapper_5/activation_1/Reluж
5module_wrapper_6/batch_normalization_1/ReadVariableOpReadVariableOp>module_wrapper_6_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype027
5module_wrapper_6/batch_normalization_1/ReadVariableOp№
7module_wrapper_6/batch_normalization_1/ReadVariableOp_1ReadVariableOp@module_wrapper_6_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7module_wrapper_6/batch_normalization_1/ReadVariableOp_1ю
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02H
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpб
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1В
7module_wrapper_6/batch_normalization_1/FusedBatchNormV3FusedBatchNormV30module_wrapper_5/activation_1/Relu:activations:0=module_wrapper_6/batch_normalization_1/ReadVariableOp:value:0?module_wrapper_6/batch_normalization_1/ReadVariableOp_1:value:0Nmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Pmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !!@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<29
7module_wrapper_6/batch_normalization_1/FusedBatchNormV3Ё
5module_wrapper_6/batch_normalization_1/AssignNewValueAssignVariableOpOmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceDmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3:batch_mean:0G^module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype027
5module_wrapper_6/batch_normalization_1/AssignNewValueЉ
7module_wrapper_6/batch_normalization_1/AssignNewValue_1AssignVariableOpQmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceHmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3:batch_variance:0I^module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype029
7module_wrapper_6/batch_normalization_1/AssignNewValue_1Ѕ
(module_wrapper_7/max_pooling2d_1/MaxPoolMaxPool;module_wrapper_6/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2*
(module_wrapper_7/max_pooling2d_1/MaxPoolс
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_8_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype021
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpю
 module_wrapper_8/conv2d_2/Conv2DConv2D1module_wrapper_7/max_pooling2d_1/MaxPool:output:07module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2"
 module_wrapper_8/conv2d_2/Conv2D┌
0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_8_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp­
!module_wrapper_8/conv2d_2/BiasAddBiasAdd)module_wrapper_8/conv2d_2/Conv2D:output:08module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2#
!module_wrapper_8/conv2d_2/BiasAddХ
"module_wrapper_9/activation_2/ReluRelu*module_wrapper_8/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2$
"module_wrapper_9/activation_2/ReluВ
6module_wrapper_10/batch_normalization_2/ReadVariableOpReadVariableOp?module_wrapper_10_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype028
6module_wrapper_10/batch_normalization_2/ReadVariableOpЫ
8module_wrapper_10/batch_normalization_2/ReadVariableOp_1ReadVariableOpAmodule_wrapper_10_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8module_wrapper_10/batch_normalization_2/ReadVariableOp_1Ъ
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ы
8module_wrapper_10/batch_normalization_2/FusedBatchNormV3FusedBatchNormV30module_wrapper_9/activation_2/Relu:activations:0>module_wrapper_10/batch_normalization_2/ReadVariableOp:value:0@module_wrapper_10/batch_normalization_2/ReadVariableOp_1:value:0Omodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Qmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2:
8module_wrapper_10/batch_normalization_2/FusedBatchNormV3і
6module_wrapper_10/batch_normalization_2/AssignNewValueAssignVariableOpPmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceEmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3:batch_mean:0H^module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype028
6module_wrapper_10/batch_normalization_2/AssignNewValueќ
8module_wrapper_10/batch_normalization_2/AssignNewValue_1AssignVariableOpRmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceImodule_wrapper_10/batch_normalization_2/FusedBatchNormV3:batch_variance:0J^module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02:
8module_wrapper_10/batch_normalization_2/AssignNewValue_1ї
)module_wrapper_11/max_pooling2d_2/MaxPoolMaxPool<module_wrapper_10/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2+
)module_wrapper_11/max_pooling2d_2/MaxPoolу
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpReadVariableOp9module_wrapper_12_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype022
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpА
!module_wrapper_12/conv2d_3/Conv2DConv2D2module_wrapper_11/max_pooling2d_2/MaxPool:output:08module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2#
!module_wrapper_12/conv2d_3/Conv2Dя
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp:module_wrapper_12_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpш
"module_wrapper_12/conv2d_3/BiasAddBiasAdd*module_wrapper_12/conv2d_3/Conv2D:output:09module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2$
"module_wrapper_12/conv2d_3/BiasAdd║
#module_wrapper_13/activation_3/ReluRelu+module_wrapper_12/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2%
#module_wrapper_13/activation_3/Reluь
6module_wrapper_14/batch_normalization_3/ReadVariableOpReadVariableOp?module_wrapper_14_batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6module_wrapper_14/batch_normalization_3/ReadVariableOpз
8module_wrapper_14/batch_normalization_3/ReadVariableOp_1ReadVariableOpAmodule_wrapper_14_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02:
8module_wrapper_14/batch_normalization_3/ReadVariableOp_1а
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02I
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpд
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02K
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Э
8module_wrapper_14/batch_normalization_3/FusedBatchNormV3FusedBatchNormV31module_wrapper_13/activation_3/Relu:activations:0>module_wrapper_14/batch_normalization_3/ReadVariableOp:value:0@module_wrapper_14/batch_normalization_3/ReadVariableOp_1:value:0Omodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Qmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2:
8module_wrapper_14/batch_normalization_3/FusedBatchNormV3і
6module_wrapper_14/batch_normalization_3/AssignNewValueAssignVariableOpPmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceEmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3:batch_mean:0H^module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype028
6module_wrapper_14/batch_normalization_3/AssignNewValueќ
8module_wrapper_14/batch_normalization_3/AssignNewValue_1AssignVariableOpRmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceImodule_wrapper_14/batch_normalization_3/FusedBatchNormV3:batch_variance:0J^module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02:
8module_wrapper_14/batch_normalization_3/AssignNewValue_1Ї
)module_wrapper_15/max_pooling2d_3/MaxPoolMaxPool<module_wrapper_14/batch_normalization_3/FusedBatchNormV3:y:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2+
)module_wrapper_15/max_pooling2d_3/MaxPoolЌ
'module_wrapper_16/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2)
'module_wrapper_16/dropout/dropout/ConstШ
%module_wrapper_16/dropout/dropout/MulMul2module_wrapper_15/max_pooling2d_3/MaxPool:output:00module_wrapper_16/dropout/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2'
%module_wrapper_16/dropout/dropout/Mul┤
'module_wrapper_16/dropout/dropout/ShapeShape2module_wrapper_15/max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:2)
'module_wrapper_16/dropout/dropout/ShapeІ
>module_wrapper_16/dropout/dropout/random_uniform/RandomUniformRandomUniform0module_wrapper_16/dropout/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02@
>module_wrapper_16/dropout/dropout/random_uniform/RandomUniformЕ
0module_wrapper_16/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>22
0module_wrapper_16/dropout/dropout/GreaterEqual/y»
.module_wrapper_16/dropout/dropout/GreaterEqualGreaterEqualGmodule_wrapper_16/dropout/dropout/random_uniform/RandomUniform:output:09module_wrapper_16/dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ20
.module_wrapper_16/dropout/dropout/GreaterEqualо
&module_wrapper_16/dropout/dropout/CastCast2module_wrapper_16/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2(
&module_wrapper_16/dropout/dropout/Castв
'module_wrapper_16/dropout/dropout/Mul_1Mul)module_wrapper_16/dropout/dropout/Mul:z:0*module_wrapper_16/dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2)
'module_wrapper_16/dropout/dropout/Mul_1Њ
module_wrapper_17/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
module_wrapper_17/flatten/Const█
!module_wrapper_17/flatten/ReshapeReshape+module_wrapper_16/dropout/dropout/Mul_1:z:0(module_wrapper_17/flatten/Const:output:0*
T0*(
_output_shapes
:         ђ2#
!module_wrapper_17/flatten/ReshapeО
-module_wrapper_18/dense/MatMul/ReadVariableOpReadVariableOp6module_wrapper_18_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02/
-module_wrapper_18/dense/MatMul/ReadVariableOpЯ
module_wrapper_18/dense/MatMulMatMul*module_wrapper_17/flatten/Reshape:output:05module_wrapper_18/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2 
module_wrapper_18/dense/MatMulН
.module_wrapper_18/dense/BiasAdd/ReadVariableOpReadVariableOp7module_wrapper_18_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.module_wrapper_18/dense/BiasAdd/ReadVariableOpР
module_wrapper_18/dense/BiasAddBiasAdd(module_wrapper_18/dense/MatMul:product:06module_wrapper_18/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2!
module_wrapper_18/dense/BiasAdd»
#module_wrapper_19/activation_4/ReluRelu(module_wrapper_18/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2%
#module_wrapper_19/activation_4/Relu┌
Fmodule_wrapper_20/batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fmodule_wrapper_20/batch_normalization_4/moments/mean/reduction_indices│
4module_wrapper_20/batch_normalization_4/moments/meanMean1module_wrapper_19/activation_4/Relu:activations:0Omodule_wrapper_20/batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(26
4module_wrapper_20/batch_normalization_4/moments/meanш
<module_wrapper_20/batch_normalization_4/moments/StopGradientStopGradient=module_wrapper_20/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	ђ2>
<module_wrapper_20/batch_normalization_4/moments/StopGradient╚
Amodule_wrapper_20/batch_normalization_4/moments/SquaredDifferenceSquaredDifference1module_wrapper_19/activation_4/Relu:activations:0Emodule_wrapper_20/batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2C
Amodule_wrapper_20/batch_normalization_4/moments/SquaredDifferenceР
Jmodule_wrapper_20/batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2L
Jmodule_wrapper_20/batch_normalization_4/moments/variance/reduction_indicesМ
8module_wrapper_20/batch_normalization_4/moments/varianceMeanEmodule_wrapper_20/batch_normalization_4/moments/SquaredDifference:z:0Smodule_wrapper_20/batch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2:
8module_wrapper_20/batch_normalization_4/moments/varianceщ
7module_wrapper_20/batch_normalization_4/moments/SqueezeSqueeze=module_wrapper_20/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 29
7module_wrapper_20/batch_normalization_4/moments/SqueezeЂ
9module_wrapper_20/batch_normalization_4/moments/Squeeze_1SqueezeAmodule_wrapper_20/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2;
9module_wrapper_20/batch_normalization_4/moments/Squeeze_1├
=module_wrapper_20/batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2?
=module_wrapper_20/batch_normalization_4/AssignMovingAvg/decayЮ
Fmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOpOmodule_wrapper_20_batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02H
Fmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOp╣
;module_wrapper_20/batch_normalization_4/AssignMovingAvg/subSubNmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0@module_wrapper_20/batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2=
;module_wrapper_20/batch_normalization_4/AssignMovingAvg/sub░
;module_wrapper_20/batch_normalization_4/AssignMovingAvg/mulMul?module_wrapper_20/batch_normalization_4/AssignMovingAvg/sub:z:0Fmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2=
;module_wrapper_20/batch_normalization_4/AssignMovingAvg/mulЄ
7module_wrapper_20/batch_normalization_4/AssignMovingAvgAssignSubVariableOpOmodule_wrapper_20_batch_normalization_4_assignmovingavg_readvariableop_resource?module_wrapper_20/batch_normalization_4/AssignMovingAvg/mul:z:0G^module_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype029
7module_wrapper_20/batch_normalization_4/AssignMovingAvgК
?module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2A
?module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/decayБ
Hmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOpQmodule_wrapper_20_batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02J
Hmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp┴
=module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/subSubPmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:0Bmodule_wrapper_20/batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2?
=module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/subИ
=module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/mulMulAmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/sub:z:0Hmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2?
=module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/mulЉ
9module_wrapper_20/batch_normalization_4/AssignMovingAvg_1AssignSubVariableOpQmodule_wrapper_20_batch_normalization_4_assignmovingavg_1_readvariableop_resourceAmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/mul:z:0I^module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02;
9module_wrapper_20/batch_normalization_4/AssignMovingAvg_1Ч
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpReadVariableOpDmodule_wrapper_20_batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02=
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpѓ
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpReadVariableOpFmodule_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpи
7module_wrapper_20/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:29
7module_wrapper_20/batch_normalization_4/batchnorm/add/yБ
5module_wrapper_20/batch_normalization_4/batchnorm/addAddV2Bmodule_wrapper_20/batch_normalization_4/moments/Squeeze_1:output:0@module_wrapper_20/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ27
5module_wrapper_20/batch_normalization_4/batchnorm/add▄
7module_wrapper_20/batch_normalization_4/batchnorm/RsqrtRsqrt9module_wrapper_20/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ29
7module_wrapper_20/batch_normalization_4/batchnorm/RsqrtЪ
5module_wrapper_20/batch_normalization_4/batchnorm/mulMul;module_wrapper_20/batch_normalization_4/batchnorm/Rsqrt:y:0Emodule_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ27
5module_wrapper_20/batch_normalization_4/batchnorm/mulџ
7module_wrapper_20/batch_normalization_4/batchnorm/mul_1Mul1module_wrapper_19/activation_4/Relu:activations:09module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ29
7module_wrapper_20/batch_normalization_4/batchnorm/mul_1ю
7module_wrapper_20/batch_normalization_4/batchnorm/mul_2Mul@module_wrapper_20/batch_normalization_4/moments/Squeeze:output:09module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ29
7module_wrapper_20/batch_normalization_4/batchnorm/mul_2Ю
5module_wrapper_20/batch_normalization_4/batchnorm/subSubCmodule_wrapper_20/batch_normalization_4/Cast/ReadVariableOp:value:0;module_wrapper_20/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ27
5module_wrapper_20/batch_normalization_4/batchnorm/subд
7module_wrapper_20/batch_normalization_4/batchnorm/add_1AddV2;module_wrapper_20/batch_normalization_4/batchnorm/mul_1:z:09module_wrapper_20/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ29
7module_wrapper_20/batch_normalization_4/batchnorm/add_1Џ
)module_wrapper_21/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)module_wrapper_21/dropout_1/dropout/Const§
'module_wrapper_21/dropout_1/dropout/MulMul;module_wrapper_20/batch_normalization_4/batchnorm/add_1:z:02module_wrapper_21/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2)
'module_wrapper_21/dropout_1/dropout/Mul┴
)module_wrapper_21/dropout_1/dropout/ShapeShape;module_wrapper_20/batch_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2+
)module_wrapper_21/dropout_1/dropout/ShapeЅ
@module_wrapper_21/dropout_1/dropout/random_uniform/RandomUniformRandomUniform2module_wrapper_21/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02B
@module_wrapper_21/dropout_1/dropout/random_uniform/RandomUniformГ
2module_wrapper_21/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2module_wrapper_21/dropout_1/dropout/GreaterEqual/y»
0module_wrapper_21/dropout_1/dropout/GreaterEqualGreaterEqualImodule_wrapper_21/dropout_1/dropout/random_uniform/RandomUniform:output:0;module_wrapper_21/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ22
0module_wrapper_21/dropout_1/dropout/GreaterEqualн
(module_wrapper_21/dropout_1/dropout/CastCast4module_wrapper_21/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2*
(module_wrapper_21/dropout_1/dropout/Castв
)module_wrapper_21/dropout_1/dropout/Mul_1Mul+module_wrapper_21/dropout_1/dropout/Mul:z:0,module_wrapper_21/dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2+
)module_wrapper_21/dropout_1/dropout/Mul_1▄
/module_wrapper_22/dense_1/MatMul/ReadVariableOpReadVariableOp8module_wrapper_22_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype021
/module_wrapper_22/dense_1/MatMul/ReadVariableOpУ
 module_wrapper_22/dense_1/MatMulMatMul-module_wrapper_21/dropout_1/dropout/Mul_1:z:07module_wrapper_22/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 module_wrapper_22/dense_1/MatMul┌
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_22_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpж
!module_wrapper_22/dense_1/BiasAddBiasAdd*module_wrapper_22/dense_1/MatMul:product:08module_wrapper_22/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2#
!module_wrapper_22/dense_1/BiasAdd╣
&module_wrapper_23/activation_5/SoftmaxSoftmax*module_wrapper_22/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2(
&module_wrapper_23/activation_5/Softmax└
IdentityIdentity0module_wrapper_23/activation_5/Softmax:softmax:0-^module_wrapper/conv2d/BiasAdd/ReadVariableOp,^module_wrapper/conv2d/Conv2D/ReadVariableOp7^module_wrapper_10/batch_normalization_2/AssignNewValue9^module_wrapper_10/batch_normalization_2/AssignNewValue_1H^module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpJ^module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17^module_wrapper_10/batch_normalization_2/ReadVariableOp9^module_wrapper_10/batch_normalization_2/ReadVariableOp_12^module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp1^module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp7^module_wrapper_14/batch_normalization_3/AssignNewValue9^module_wrapper_14/batch_normalization_3/AssignNewValue_1H^module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpJ^module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17^module_wrapper_14/batch_normalization_3/ReadVariableOp9^module_wrapper_14/batch_normalization_3/ReadVariableOp_1/^module_wrapper_18/dense/BiasAdd/ReadVariableOp.^module_wrapper_18/dense/MatMul/ReadVariableOp4^module_wrapper_2/batch_normalization/AssignNewValue6^module_wrapper_2/batch_normalization/AssignNewValue_1E^module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpG^module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_14^module_wrapper_2/batch_normalization/ReadVariableOp6^module_wrapper_2/batch_normalization/ReadVariableOp_18^module_wrapper_20/batch_normalization_4/AssignMovingAvgG^module_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOp:^module_wrapper_20/batch_normalization_4/AssignMovingAvg_1I^module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp<^module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp>^module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp1^module_wrapper_22/dense_1/BiasAdd/ReadVariableOp0^module_wrapper_22/dense_1/MatMul/ReadVariableOp1^module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp0^module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp6^module_wrapper_6/batch_normalization_1/AssignNewValue8^module_wrapper_6/batch_normalization_1/AssignNewValue_1G^module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpI^module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_16^module_wrapper_6/batch_normalization_1/ReadVariableOp8^module_wrapper_6/batch_normalization_1/ReadVariableOp_11^module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp0^module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,module_wrapper/conv2d/BiasAdd/ReadVariableOp,module_wrapper/conv2d/BiasAdd/ReadVariableOp2Z
+module_wrapper/conv2d/Conv2D/ReadVariableOp+module_wrapper/conv2d/Conv2D/ReadVariableOp2p
6module_wrapper_10/batch_normalization_2/AssignNewValue6module_wrapper_10/batch_normalization_2/AssignNewValue2t
8module_wrapper_10/batch_normalization_2/AssignNewValue_18module_wrapper_10/batch_normalization_2/AssignNewValue_12њ
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpGmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2ќ
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12p
6module_wrapper_10/batch_normalization_2/ReadVariableOp6module_wrapper_10/batch_normalization_2/ReadVariableOp2t
8module_wrapper_10/batch_normalization_2/ReadVariableOp_18module_wrapper_10/batch_normalization_2/ReadVariableOp_12f
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp2d
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp2p
6module_wrapper_14/batch_normalization_3/AssignNewValue6module_wrapper_14/batch_normalization_3/AssignNewValue2t
8module_wrapper_14/batch_normalization_3/AssignNewValue_18module_wrapper_14/batch_normalization_3/AssignNewValue_12њ
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpGmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2ќ
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12p
6module_wrapper_14/batch_normalization_3/ReadVariableOp6module_wrapper_14/batch_normalization_3/ReadVariableOp2t
8module_wrapper_14/batch_normalization_3/ReadVariableOp_18module_wrapper_14/batch_normalization_3/ReadVariableOp_12`
.module_wrapper_18/dense/BiasAdd/ReadVariableOp.module_wrapper_18/dense/BiasAdd/ReadVariableOp2^
-module_wrapper_18/dense/MatMul/ReadVariableOp-module_wrapper_18/dense/MatMul/ReadVariableOp2j
3module_wrapper_2/batch_normalization/AssignNewValue3module_wrapper_2/batch_normalization/AssignNewValue2n
5module_wrapper_2/batch_normalization/AssignNewValue_15module_wrapper_2/batch_normalization/AssignNewValue_12ї
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpDmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp2љ
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3module_wrapper_2/batch_normalization/ReadVariableOp3module_wrapper_2/batch_normalization/ReadVariableOp2n
5module_wrapper_2/batch_normalization/ReadVariableOp_15module_wrapper_2/batch_normalization/ReadVariableOp_12r
7module_wrapper_20/batch_normalization_4/AssignMovingAvg7module_wrapper_20/batch_normalization_4/AssignMovingAvg2љ
Fmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOpFmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOp2v
9module_wrapper_20/batch_normalization_4/AssignMovingAvg_19module_wrapper_20/batch_normalization_4/AssignMovingAvg_12ћ
Hmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpHmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2z
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp2~
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp2d
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOp0module_wrapper_22/dense_1/BiasAdd/ReadVariableOp2b
/module_wrapper_22/dense_1/MatMul/ReadVariableOp/module_wrapper_22/dense_1/MatMul/ReadVariableOp2d
0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp2b
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp2n
5module_wrapper_6/batch_normalization_1/AssignNewValue5module_wrapper_6/batch_normalization_1/AssignNewValue2r
7module_wrapper_6/batch_normalization_1/AssignNewValue_17module_wrapper_6/batch_normalization_1/AssignNewValue_12љ
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpFmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2ћ
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5module_wrapper_6/batch_normalization_1/ReadVariableOp5module_wrapper_6/batch_normalization_1/ReadVariableOp2r
7module_wrapper_6/batch_normalization_1/ReadVariableOp_17module_wrapper_6/batch_normalization_1/ReadVariableOp_12d
0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp2b
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Ј
h
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_56906

args_0
identityi
activation_4/ReluReluargs_0*
T0*(
_output_shapes
:         ђ2
activation_4/Relut
IdentityIdentityactivation_4/Relu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
ф
g
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_57430

args_0
identityp
activation_2/ReluReluargs_0*
T0*/
_output_shapes
:         @2
activation_2/Relu{
IdentityIdentityactivation_2/Relu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
А
h
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_57224

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/Constђ
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
­
M
1__inference_module_wrapper_16_layer_call_fn_59947

args_0
identityМ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_568752
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ћ
╠
1__inference_module_wrapper_10_layer_call_fn_59765

args_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_574062
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Я
M
1__inference_module_wrapper_17_layer_call_fn_59969

args_0
identity╦
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_568832
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
л
M
1__inference_module_wrapper_19_layer_call_fn_60027

args_0
identity╦
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_569062
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
»
h
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_59833

args_0
identityq
activation_3/ReluReluargs_0*
T0*0
_output_shapes
:         ђ2
activation_3/Relu|
IdentityIdentityactivation_3/Relu:activations:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Т
Є
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_58809

inputs+
cast_readvariableop_resource:	ђ-
cast_1_readvariableop_resource:	ђ-
cast_2_readvariableop_resource:	ђ-
cast_3_readvariableop_resource:	ђ
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Н
K
/__inference_max_pooling2d_1_layer_call_fn_58509

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_585032
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ј
е
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_57455

args_0A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@
identityѕбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOp░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpЙ
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_2/Conv2DД
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpг
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAddИ
IdentityIdentityconv2d_2/BiasAdd:output:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
у
л
5__inference_batch_normalization_2_layer_call_fn_60370

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_585312
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ж
L
0__inference_module_wrapper_1_layer_call_fn_59423

args_0
identityЛ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_576422
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
ф
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_58779

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Й
┐
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_58437

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Л
Ц
0__inference_module_wrapper_4_layer_call_fn_59534

args_0!
unknown: @
	unknown_0:@
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_567062
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         !! : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameargs_0
Н
K
/__inference_max_pooling2d_2_layer_call_fn_58647

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_586412
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ќ
Ф
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_59805

args_0B
'conv2d_3_conv2d_readvariableop_resource:@ђ7
(conv2d_3_biasadd_readvariableop_resource:	ђ
identityѕбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOp▒
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_3/Conv2D/ReadVariableOp┐
conv2d_3/Conv2DConv2Dargs_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_3/Conv2Dе
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpГ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_3/BiasAdd╣
IdentityIdentityconv2d_3/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
╠
M
1__inference_module_wrapper_23_layer_call_fn_60197

args_0
identity╩
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_570492
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameargs_0
«
Ъ
1__inference_module_wrapper_22_layer_call_fn_60168

args_0
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_569552
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ж
L
0__inference_module_wrapper_5_layer_call_fn_59563

args_0
identityЛ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_575362
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
Ч
j
1__inference_module_wrapper_16_layer_call_fn_59952

args_0
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_572472
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Н
K
/__inference_max_pooling2d_3_layer_call_fn_58785

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_587792
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ф
g
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_59553

args_0
identityp
activation_1/ReluReluargs_0*
T0*/
_output_shapes
:         !!@2
activation_1/Relu{
IdentityIdentityactivation_1/Relu:activations:0*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
┘
ќ
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_59599

args_0;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб$batch_normalization_1/AssignNewValueб&batch_normalization_1/AssignNewValue_1б5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1Х
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1ж
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1▄
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !!@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_1/FusedBatchNormV3░
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue╝
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1ў
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         !!@: : : : 2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_1:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
Ј
е
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_59665

args_0A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@
identityѕбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOp░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpЙ
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_2/Conv2DД
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpг
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAddИ
IdentityIdentityconv2d_2/BiasAdd:output:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
­
M
1__inference_module_wrapper_13_layer_call_fn_59843

args_0
identityМ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_573242
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
┘
ќ
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_57512

args_0;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб$batch_normalization_1/AssignNewValueб&batch_normalization_1/AssignNewValue_1б5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1Х
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1ж
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1▄
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !!@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_1/FusedBatchNormV3░
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue╝
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1ў
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         !!@: : : : 2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_1:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
═
Б
.__inference_module_wrapper_layer_call_fn_59394

args_0!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_566482
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameargs_0
у
ѓ
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_57618

args_09
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб"batch_normalization/AssignNewValueб$batch_normalization/AssignNewValue_1б3batch_normalization/FusedBatchNormV3/ReadVariableOpб5batch_normalization/FusedBatchNormV3/ReadVariableOp_1б"batch_normalization/ReadVariableOpб$batch_normalization/ReadVariableOp_1░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpХ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1с
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpж
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1л
$batch_normalization/FusedBatchNormV3FusedBatchNormV3args_0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<2&
$batch_normalization/FusedBatchNormV3д
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue▓
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1і
IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         dd : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_1:W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
П
ъ
I__inference_module_wrapper_layer_call_and_return_conditional_losses_56648

args_0?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpИ
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd 2
conv2d/BiasAdd▓
IdentityIdentityconv2d/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameargs_0
А
h
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_56883

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/Constђ
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
╔
▄
*__inference_sequential_layer_call_fn_59296

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@ђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ

unknown_21:	ђ

unknown_22:	ђ

unknown_23:
ђђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ

unknown_27:	ђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:
identityѕбStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_569692
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
еЄ
у)
 __inference__wrapped_model_56631
module_wrapper_inputY
?sequential_module_wrapper_conv2d_conv2d_readvariableop_resource: N
@sequential_module_wrapper_conv2d_biasadd_readvariableop_resource: U
Gsequential_module_wrapper_2_batch_normalization_readvariableop_resource: W
Isequential_module_wrapper_2_batch_normalization_readvariableop_1_resource: f
Xsequential_module_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resource: h
Zsequential_module_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: ]
Csequential_module_wrapper_4_conv2d_1_conv2d_readvariableop_resource: @R
Dsequential_module_wrapper_4_conv2d_1_biasadd_readvariableop_resource:@W
Isequential_module_wrapper_6_batch_normalization_1_readvariableop_resource:@Y
Ksequential_module_wrapper_6_batch_normalization_1_readvariableop_1_resource:@h
Zsequential_module_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@j
\sequential_module_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@]
Csequential_module_wrapper_8_conv2d_2_conv2d_readvariableop_resource:@@R
Dsequential_module_wrapper_8_conv2d_2_biasadd_readvariableop_resource:@X
Jsequential_module_wrapper_10_batch_normalization_2_readvariableop_resource:@Z
Lsequential_module_wrapper_10_batch_normalization_2_readvariableop_1_resource:@i
[sequential_module_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@k
]sequential_module_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@_
Dsequential_module_wrapper_12_conv2d_3_conv2d_readvariableop_resource:@ђT
Esequential_module_wrapper_12_conv2d_3_biasadd_readvariableop_resource:	ђY
Jsequential_module_wrapper_14_batch_normalization_3_readvariableop_resource:	ђ[
Lsequential_module_wrapper_14_batch_normalization_3_readvariableop_1_resource:	ђj
[sequential_module_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђl
]sequential_module_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђU
Asequential_module_wrapper_18_dense_matmul_readvariableop_resource:
ђђQ
Bsequential_module_wrapper_18_dense_biasadd_readvariableop_resource:	ђ^
Osequential_module_wrapper_20_batch_normalization_4_cast_readvariableop_resource:	ђ`
Qsequential_module_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource:	ђ`
Qsequential_module_wrapper_20_batch_normalization_4_cast_2_readvariableop_resource:	ђ`
Qsequential_module_wrapper_20_batch_normalization_4_cast_3_readvariableop_resource:	ђV
Csequential_module_wrapper_22_dense_1_matmul_readvariableop_resource:	ђR
Dsequential_module_wrapper_22_dense_1_biasadd_readvariableop_resource:
identityѕб7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOpб6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOpбRsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбTsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1бAsequential/module_wrapper_10/batch_normalization_2/ReadVariableOpбCsequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1б<sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpб;sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpбRsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбTsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1бAsequential/module_wrapper_14/batch_normalization_3/ReadVariableOpбCsequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1б9sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOpб8sequential/module_wrapper_18/dense/MatMul/ReadVariableOpбOsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpбQsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1б>sequential/module_wrapper_2/batch_normalization/ReadVariableOpб@sequential/module_wrapper_2/batch_normalization/ReadVariableOp_1бFsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpбHsequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpбHsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpбHsequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpб;sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOpб:sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOpб;sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpб:sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpбQsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбSsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б@sequential/module_wrapper_6/batch_normalization_1/ReadVariableOpбBsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1б;sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpб:sequential/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpЭ
6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOpReadVariableOp?sequential_module_wrapper_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype028
6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOpћ
'sequential/module_wrapper/conv2d/Conv2DConv2Dmodule_wrapper_input>sequential/module_wrapper/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2)
'sequential/module_wrapper/conv2d/Conv2D№
7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOpReadVariableOp@sequential_module_wrapper_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOpї
(sequential/module_wrapper/conv2d/BiasAddBiasAdd0sequential/module_wrapper/conv2d/Conv2D:output:0?sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd 2*
(sequential/module_wrapper/conv2d/BiasAdd¤
+sequential/module_wrapper_1/activation/ReluRelu1sequential/module_wrapper/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         dd 2-
+sequential/module_wrapper_1/activation/Reluё
>sequential/module_wrapper_2/batch_normalization/ReadVariableOpReadVariableOpGsequential_module_wrapper_2_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02@
>sequential/module_wrapper_2/batch_normalization/ReadVariableOpі
@sequential/module_wrapper_2/batch_normalization/ReadVariableOp_1ReadVariableOpIsequential_module_wrapper_2_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@sequential/module_wrapper_2/batch_normalization/ReadVariableOp_1и
Osequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpXsequential_module_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Q
Osequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpй
Qsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZsequential_module_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02S
Qsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ю
@sequential/module_wrapper_2/batch_normalization/FusedBatchNormV3FusedBatchNormV39sequential/module_wrapper_1/activation/Relu:activations:0Fsequential/module_wrapper_2/batch_normalization/ReadVariableOp:value:0Hsequential/module_wrapper_2/batch_normalization/ReadVariableOp_1:value:0Wsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ysequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oЃ:*
is_training( 2B
@sequential/module_wrapper_2/batch_normalization/FusedBatchNormV3ц
1sequential/module_wrapper_3/max_pooling2d/MaxPoolMaxPoolDsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
23
1sequential/module_wrapper_3/max_pooling2d/MaxPoolё
:sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpReadVariableOpCsequential_module_wrapper_4_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02<
:sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpк
+sequential/module_wrapper_4/conv2d_1/Conv2DConv2D:sequential/module_wrapper_3/max_pooling2d/MaxPool:output:0Bsequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2-
+sequential/module_wrapper_4/conv2d_1/Conv2Dч
;sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpDsequential_module_wrapper_4_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpю
,sequential/module_wrapper_4/conv2d_1/BiasAddBiasAdd4sequential/module_wrapper_4/conv2d_1/Conv2D:output:0Csequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2.
,sequential/module_wrapper_4/conv2d_1/BiasAddО
-sequential/module_wrapper_5/activation_1/ReluRelu5sequential/module_wrapper_4/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         !!@2/
-sequential/module_wrapper_5/activation_1/Reluі
@sequential/module_wrapper_6/batch_normalization_1/ReadVariableOpReadVariableOpIsequential_module_wrapper_6_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@sequential/module_wrapper_6/batch_normalization_1/ReadVariableOpљ
Bsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1ReadVariableOpKsequential_module_wrapper_6_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1й
Qsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpZsequential_module_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02S
Qsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp├
Ssequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\sequential_module_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02U
Ssequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ф
Bsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3;sequential/module_wrapper_5/activation_1/Relu:activations:0Hsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp:value:0Jsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1:value:0Ysequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0[sequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !!@:@:@:@:@:*
epsilon%oЃ:*
is_training( 2D
Bsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3ф
3sequential/module_wrapper_7/max_pooling2d_1/MaxPoolMaxPoolFsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
25
3sequential/module_wrapper_7/max_pooling2d_1/MaxPoolё
:sequential/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpReadVariableOpCsequential_module_wrapper_8_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:sequential/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp╚
+sequential/module_wrapper_8/conv2d_2/Conv2DConv2D<sequential/module_wrapper_7/max_pooling2d_1/MaxPool:output:0Bsequential/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2-
+sequential/module_wrapper_8/conv2d_2/Conv2Dч
;sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpDsequential_module_wrapper_8_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpю
,sequential/module_wrapper_8/conv2d_2/BiasAddBiasAdd4sequential/module_wrapper_8/conv2d_2/Conv2D:output:0Csequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2.
,sequential/module_wrapper_8/conv2d_2/BiasAddО
-sequential/module_wrapper_9/activation_2/ReluRelu5sequential/module_wrapper_8/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2/
-sequential/module_wrapper_9/activation_2/ReluЇ
Asequential/module_wrapper_10/batch_normalization_2/ReadVariableOpReadVariableOpJsequential_module_wrapper_10_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02C
Asequential/module_wrapper_10/batch_normalization_2/ReadVariableOpЊ
Csequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1ReadVariableOpLsequential_module_wrapper_10_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Csequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1└
Rsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp[sequential_module_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpк
Tsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]sequential_module_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02V
Tsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1▒
Csequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3;sequential/module_wrapper_9/activation_2/Relu:activations:0Isequential/module_wrapper_10/batch_normalization_2/ReadVariableOp:value:0Ksequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1:value:0Zsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0\sequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2E
Csequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3Г
4sequential/module_wrapper_11/max_pooling2d_2/MaxPoolMaxPoolGsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
26
4sequential/module_wrapper_11/max_pooling2d_2/MaxPoolѕ
;sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpReadVariableOpDsequential_module_wrapper_12_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02=
;sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp═
,sequential/module_wrapper_12/conv2d_3/Conv2DConv2D=sequential/module_wrapper_11/max_pooling2d_2/MaxPool:output:0Csequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2.
,sequential/module_wrapper_12/conv2d_3/Conv2D 
<sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpEsequential_module_wrapper_12_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02>
<sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpА
-sequential/module_wrapper_12/conv2d_3/BiasAddBiasAdd5sequential/module_wrapper_12/conv2d_3/Conv2D:output:0Dsequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2/
-sequential/module_wrapper_12/conv2d_3/BiasAdd█
.sequential/module_wrapper_13/activation_3/ReluRelu6sequential/module_wrapper_12/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ20
.sequential/module_wrapper_13/activation_3/Reluј
Asequential/module_wrapper_14/batch_normalization_3/ReadVariableOpReadVariableOpJsequential_module_wrapper_14_batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02C
Asequential/module_wrapper_14/batch_normalization_3/ReadVariableOpћ
Csequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1ReadVariableOpLsequential_module_wrapper_14_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02E
Csequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1┴
Rsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp[sequential_module_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02T
Rsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpК
Tsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]sequential_module_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02V
Tsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1и
Csequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3<sequential/module_wrapper_13/activation_3/Relu:activations:0Isequential/module_wrapper_14/batch_normalization_3/ReadVariableOp:value:0Ksequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1:value:0Zsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0\sequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2E
Csequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3«
4sequential/module_wrapper_15/max_pooling2d_3/MaxPoolMaxPoolGsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3:y:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
26
4sequential/module_wrapper_15/max_pooling2d_3/MaxPoolС
-sequential/module_wrapper_16/dropout/IdentityIdentity=sequential/module_wrapper_15/max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2/
-sequential/module_wrapper_16/dropout/IdentityЕ
*sequential/module_wrapper_17/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*sequential/module_wrapper_17/flatten/ConstЄ
,sequential/module_wrapper_17/flatten/ReshapeReshape6sequential/module_wrapper_16/dropout/Identity:output:03sequential/module_wrapper_17/flatten/Const:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential/module_wrapper_17/flatten/ReshapeЭ
8sequential/module_wrapper_18/dense/MatMul/ReadVariableOpReadVariableOpAsequential_module_wrapper_18_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02:
8sequential/module_wrapper_18/dense/MatMul/ReadVariableOpї
)sequential/module_wrapper_18/dense/MatMulMatMul5sequential/module_wrapper_17/flatten/Reshape:output:0@sequential/module_wrapper_18/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2+
)sequential/module_wrapper_18/dense/MatMulШ
9sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOpReadVariableOpBsequential_module_wrapper_18_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02;
9sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOpј
*sequential/module_wrapper_18/dense/BiasAddBiasAdd3sequential/module_wrapper_18/dense/MatMul:product:0Asequential/module_wrapper_18/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2,
*sequential/module_wrapper_18/dense/BiasAddл
.sequential/module_wrapper_19/activation_4/ReluRelu3sequential/module_wrapper_18/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ20
.sequential/module_wrapper_19/activation_4/ReluЮ
Fsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpReadVariableOpOsequential_module_wrapper_20_batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02H
Fsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpБ
Hsequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpReadVariableOpQsequential_module_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02J
Hsequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpБ
Hsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpReadVariableOpQsequential_module_wrapper_20_batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02J
Hsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpБ
Hsequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpReadVariableOpQsequential_module_wrapper_20_batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02J
Hsequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp═
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2D
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/add/yм
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/addAddV2Psequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp:value:0Ksequential/module_wrapper_20/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2B
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/add§
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/RsqrtRsqrtDsequential/module_wrapper_20/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2D
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/Rsqrt╦
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/mulMulFsequential/module_wrapper_20/batch_normalization_4/batchnorm/Rsqrt:y:0Psequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2B
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/mulк
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_1Mul<sequential/module_wrapper_19/activation_4/Relu:activations:0Dsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2D
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_1╦
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_2MulNsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp:value:0Dsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2D
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_2╦
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/subSubPsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOp:value:0Fsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2B
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/subм
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/add_1AddV2Fsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_1:z:0Dsequential/module_wrapper_20/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2D
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/add_1ж
/sequential/module_wrapper_21/dropout_1/IdentityIdentityFsequential/module_wrapper_20/batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         ђ21
/sequential/module_wrapper_21/dropout_1/Identity§
:sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOpReadVariableOpCsequential_module_wrapper_22_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02<
:sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOpћ
+sequential/module_wrapper_22/dense_1/MatMulMatMul8sequential/module_wrapper_21/dropout_1/Identity:output:0Bsequential/module_wrapper_22/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2-
+sequential/module_wrapper_22/dense_1/MatMulч
;sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOpReadVariableOpDsequential_module_wrapper_22_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOpЋ
,sequential/module_wrapper_22/dense_1/BiasAddBiasAdd5sequential/module_wrapper_22/dense_1/MatMul:product:0Csequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2.
,sequential/module_wrapper_22/dense_1/BiasAdd┌
1sequential/module_wrapper_23/activation_5/SoftmaxSoftmax5sequential/module_wrapper_22/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         23
1sequential/module_wrapper_23/activation_5/Softmax┘
IdentityIdentity;sequential/module_wrapper_23/activation_5/Softmax:softmax:08^sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOp7^sequential/module_wrapper/conv2d/Conv2D/ReadVariableOpS^sequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpU^sequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1B^sequential/module_wrapper_10/batch_normalization_2/ReadVariableOpD^sequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1=^sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp<^sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpS^sequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpU^sequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1B^sequential/module_wrapper_14/batch_normalization_3/ReadVariableOpD^sequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1:^sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOp9^sequential/module_wrapper_18/dense/MatMul/ReadVariableOpP^sequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpR^sequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?^sequential/module_wrapper_2/batch_normalization/ReadVariableOpA^sequential/module_wrapper_2/batch_normalization/ReadVariableOp_1G^sequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpI^sequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpI^sequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpI^sequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp<^sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOp;^sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOp<^sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp;^sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpR^sequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpT^sequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1A^sequential/module_wrapper_6/batch_normalization_1/ReadVariableOpC^sequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1<^sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp;^sequential/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOp7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOp2p
6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOp6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOp2е
Rsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpRsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2г
Tsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Tsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12є
Asequential/module_wrapper_10/batch_normalization_2/ReadVariableOpAsequential/module_wrapper_10/batch_normalization_2/ReadVariableOp2і
Csequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1Csequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_12|
<sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp<sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp2z
;sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp;sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp2е
Rsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpRsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2г
Tsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Tsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12є
Asequential/module_wrapper_14/batch_normalization_3/ReadVariableOpAsequential/module_wrapper_14/batch_normalization_3/ReadVariableOp2і
Csequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1Csequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_12v
9sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOp9sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOp2t
8sequential/module_wrapper_18/dense/MatMul/ReadVariableOp8sequential/module_wrapper_18/dense/MatMul/ReadVariableOp2б
Osequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpOsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp2д
Qsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Qsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_12ђ
>sequential/module_wrapper_2/batch_normalization/ReadVariableOp>sequential/module_wrapper_2/batch_normalization/ReadVariableOp2ё
@sequential/module_wrapper_2/batch_normalization/ReadVariableOp_1@sequential/module_wrapper_2/batch_normalization/ReadVariableOp_12љ
Fsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpFsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp2ћ
Hsequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpHsequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp2ћ
Hsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpHsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOp2ћ
Hsequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpHsequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp2z
;sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOp;sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOp2x
:sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOp:sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOp2z
;sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp;sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp2x
:sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp:sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp2д
Qsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpQsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2ф
Ssequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ssequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12ё
@sequential/module_wrapper_6/batch_normalization_1/ReadVariableOp@sequential/module_wrapper_6/batch_normalization_1/ReadVariableOp2ѕ
Bsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1Bsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_12z
;sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp;sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp2x
:sequential/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:sequential/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:e a
/
_output_shapes
:         dd
.
_user_specified_namemodule_wrapper_input
ф
g
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_56717

args_0
identityp
activation_1/ReluReluargs_0*
T0*/
_output_shapes
:         !!@2
activation_1/Relu{
IdentityIdentityactivation_1/Relu:activations:0*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
О
е
1__inference_module_wrapper_12_layer_call_fn_59823

args_0"
unknown:@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_573492
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
╬
├
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_60419

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
О
е
1__inference_module_wrapper_12_layer_call_fn_59814

args_0"
unknown:@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_568222
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ф
h
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_59930

args_0
identitys
dropout/IdentityIdentityargs_0*
T0*0
_output_shapes
:         ђ2
dropout/Identityv
IdentityIdentitydropout/Identity:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
П
ъ
I__inference_module_wrapper_layer_call_and_return_conditional_losses_59375

args_0?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpИ
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd 2
conv2d/BiasAdd▓
IdentityIdentityconv2d/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameargs_0
Я
M
1__inference_module_wrapper_17_layer_call_fn_59974

args_0
identity╦
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_572242
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
ф
╦
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_59861

args_0<
-batch_normalization_3_readvariableop_resource:	ђ>
/batch_normalization_3_readvariableop_1_resource:	ђM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1и
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_3/ReadVariableOpй
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_3/ReadVariableOp_1Ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1М
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3╔
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ђ: : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_1:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
­
M
1__inference_module_wrapper_15_layer_call_fn_59925

args_0
identityМ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_572632
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ж
L
0__inference_module_wrapper_7_layer_call_fn_59645

args_0
identityЛ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_574752
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
№
h
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_59770

args_0
identity▓
max_pooling2d_2/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool|
IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
§
л
1__inference_module_wrapper_20_layer_call_fn_60112

args_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_571542
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
ф
g
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_59688

args_0
identityp
activation_2/ReluReluargs_0*
T0*/
_output_shapes
:         @2
activation_2/Relu{
IdentityIdentityactivation_2/Relu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ќ
Ф
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_57349

args_0B
'conv2d_3_conv2d_readvariableop_resource:@ђ7
(conv2d_3_biasadd_readvariableop_resource:	ђ
identityѕбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOp▒
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_3/Conv2D/ReadVariableOp┐
conv2d_3/Conv2DConv2Dargs_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_3/Conv2Dе
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpГ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_3/BiasAdd╣
IdentityIdentityconv2d_3/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ж
L
0__inference_module_wrapper_5_layer_call_fn_59558

args_0
identityЛ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_567172
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
┌
Ќ
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_59739

args_0;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб$batch_normalization_2/AssignNewValueб&batch_normalization_2/AssignNewValue_1б5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1Х
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1ж
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1▄
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_2/FusedBatchNormV3░
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue╝
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1ў
IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : : : 2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_1:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
юѕ
Ћ$
__inference__traced_save_60734
file_prefix'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop;
7savev2_module_wrapper_conv2d_kernel_read_readvariableop9
5savev2_module_wrapper_conv2d_bias_read_readvariableopI
Esavev2_module_wrapper_2_batch_normalization_gamma_read_readvariableopH
Dsavev2_module_wrapper_2_batch_normalization_beta_read_readvariableopO
Ksavev2_module_wrapper_2_batch_normalization_moving_mean_read_readvariableopS
Osavev2_module_wrapper_2_batch_normalization_moving_variance_read_readvariableop?
;savev2_module_wrapper_4_conv2d_1_kernel_read_readvariableop=
9savev2_module_wrapper_4_conv2d_1_bias_read_readvariableopK
Gsavev2_module_wrapper_6_batch_normalization_1_gamma_read_readvariableopJ
Fsavev2_module_wrapper_6_batch_normalization_1_beta_read_readvariableopQ
Msavev2_module_wrapper_6_batch_normalization_1_moving_mean_read_readvariableopU
Qsavev2_module_wrapper_6_batch_normalization_1_moving_variance_read_readvariableop?
;savev2_module_wrapper_8_conv2d_2_kernel_read_readvariableop=
9savev2_module_wrapper_8_conv2d_2_bias_read_readvariableopL
Hsavev2_module_wrapper_10_batch_normalization_2_gamma_read_readvariableopK
Gsavev2_module_wrapper_10_batch_normalization_2_beta_read_readvariableopR
Nsavev2_module_wrapper_10_batch_normalization_2_moving_mean_read_readvariableopV
Rsavev2_module_wrapper_10_batch_normalization_2_moving_variance_read_readvariableop@
<savev2_module_wrapper_12_conv2d_3_kernel_read_readvariableop>
:savev2_module_wrapper_12_conv2d_3_bias_read_readvariableopL
Hsavev2_module_wrapper_14_batch_normalization_3_gamma_read_readvariableopK
Gsavev2_module_wrapper_14_batch_normalization_3_beta_read_readvariableopR
Nsavev2_module_wrapper_14_batch_normalization_3_moving_mean_read_readvariableopV
Rsavev2_module_wrapper_14_batch_normalization_3_moving_variance_read_readvariableop=
9savev2_module_wrapper_18_dense_kernel_read_readvariableop;
7savev2_module_wrapper_18_dense_bias_read_readvariableopL
Hsavev2_module_wrapper_20_batch_normalization_4_gamma_read_readvariableopK
Gsavev2_module_wrapper_20_batch_normalization_4_beta_read_readvariableopR
Nsavev2_module_wrapper_20_batch_normalization_4_moving_mean_read_readvariableopV
Rsavev2_module_wrapper_20_batch_normalization_4_moving_variance_read_readvariableop?
;savev2_module_wrapper_22_dense_1_kernel_read_readvariableop=
9savev2_module_wrapper_22_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopH
Dsavev2_sgd_module_wrapper_conv2d_kernel_momentum_read_readvariableopF
Bsavev2_sgd_module_wrapper_conv2d_bias_momentum_read_readvariableopV
Rsavev2_sgd_module_wrapper_2_batch_normalization_gamma_momentum_read_readvariableopU
Qsavev2_sgd_module_wrapper_2_batch_normalization_beta_momentum_read_readvariableopL
Hsavev2_sgd_module_wrapper_4_conv2d_1_kernel_momentum_read_readvariableopJ
Fsavev2_sgd_module_wrapper_4_conv2d_1_bias_momentum_read_readvariableopX
Tsavev2_sgd_module_wrapper_6_batch_normalization_1_gamma_momentum_read_readvariableopW
Ssavev2_sgd_module_wrapper_6_batch_normalization_1_beta_momentum_read_readvariableopL
Hsavev2_sgd_module_wrapper_8_conv2d_2_kernel_momentum_read_readvariableopJ
Fsavev2_sgd_module_wrapper_8_conv2d_2_bias_momentum_read_readvariableopY
Usavev2_sgd_module_wrapper_10_batch_normalization_2_gamma_momentum_read_readvariableopX
Tsavev2_sgd_module_wrapper_10_batch_normalization_2_beta_momentum_read_readvariableopM
Isavev2_sgd_module_wrapper_12_conv2d_3_kernel_momentum_read_readvariableopK
Gsavev2_sgd_module_wrapper_12_conv2d_3_bias_momentum_read_readvariableopY
Usavev2_sgd_module_wrapper_14_batch_normalization_3_gamma_momentum_read_readvariableopX
Tsavev2_sgd_module_wrapper_14_batch_normalization_3_beta_momentum_read_readvariableopJ
Fsavev2_sgd_module_wrapper_18_dense_kernel_momentum_read_readvariableopH
Dsavev2_sgd_module_wrapper_18_dense_bias_momentum_read_readvariableopY
Usavev2_sgd_module_wrapper_20_batch_normalization_4_gamma_momentum_read_readvariableopX
Tsavev2_sgd_module_wrapper_20_batch_normalization_4_beta_momentum_read_readvariableopL
Hsavev2_sgd_module_wrapper_22_dense_1_kernel_momentum_read_readvariableopJ
Fsavev2_sgd_module_wrapper_22_dense_1_bias_momentum_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╔
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*█
valueЛB╬?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/30/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/31/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЅ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*Њ
valueЅBє?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesф#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop7savev2_module_wrapper_conv2d_kernel_read_readvariableop5savev2_module_wrapper_conv2d_bias_read_readvariableopEsavev2_module_wrapper_2_batch_normalization_gamma_read_readvariableopDsavev2_module_wrapper_2_batch_normalization_beta_read_readvariableopKsavev2_module_wrapper_2_batch_normalization_moving_mean_read_readvariableopOsavev2_module_wrapper_2_batch_normalization_moving_variance_read_readvariableop;savev2_module_wrapper_4_conv2d_1_kernel_read_readvariableop9savev2_module_wrapper_4_conv2d_1_bias_read_readvariableopGsavev2_module_wrapper_6_batch_normalization_1_gamma_read_readvariableopFsavev2_module_wrapper_6_batch_normalization_1_beta_read_readvariableopMsavev2_module_wrapper_6_batch_normalization_1_moving_mean_read_readvariableopQsavev2_module_wrapper_6_batch_normalization_1_moving_variance_read_readvariableop;savev2_module_wrapper_8_conv2d_2_kernel_read_readvariableop9savev2_module_wrapper_8_conv2d_2_bias_read_readvariableopHsavev2_module_wrapper_10_batch_normalization_2_gamma_read_readvariableopGsavev2_module_wrapper_10_batch_normalization_2_beta_read_readvariableopNsavev2_module_wrapper_10_batch_normalization_2_moving_mean_read_readvariableopRsavev2_module_wrapper_10_batch_normalization_2_moving_variance_read_readvariableop<savev2_module_wrapper_12_conv2d_3_kernel_read_readvariableop:savev2_module_wrapper_12_conv2d_3_bias_read_readvariableopHsavev2_module_wrapper_14_batch_normalization_3_gamma_read_readvariableopGsavev2_module_wrapper_14_batch_normalization_3_beta_read_readvariableopNsavev2_module_wrapper_14_batch_normalization_3_moving_mean_read_readvariableopRsavev2_module_wrapper_14_batch_normalization_3_moving_variance_read_readvariableop9savev2_module_wrapper_18_dense_kernel_read_readvariableop7savev2_module_wrapper_18_dense_bias_read_readvariableopHsavev2_module_wrapper_20_batch_normalization_4_gamma_read_readvariableopGsavev2_module_wrapper_20_batch_normalization_4_beta_read_readvariableopNsavev2_module_wrapper_20_batch_normalization_4_moving_mean_read_readvariableopRsavev2_module_wrapper_20_batch_normalization_4_moving_variance_read_readvariableop;savev2_module_wrapper_22_dense_1_kernel_read_readvariableop9savev2_module_wrapper_22_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopDsavev2_sgd_module_wrapper_conv2d_kernel_momentum_read_readvariableopBsavev2_sgd_module_wrapper_conv2d_bias_momentum_read_readvariableopRsavev2_sgd_module_wrapper_2_batch_normalization_gamma_momentum_read_readvariableopQsavev2_sgd_module_wrapper_2_batch_normalization_beta_momentum_read_readvariableopHsavev2_sgd_module_wrapper_4_conv2d_1_kernel_momentum_read_readvariableopFsavev2_sgd_module_wrapper_4_conv2d_1_bias_momentum_read_readvariableopTsavev2_sgd_module_wrapper_6_batch_normalization_1_gamma_momentum_read_readvariableopSsavev2_sgd_module_wrapper_6_batch_normalization_1_beta_momentum_read_readvariableopHsavev2_sgd_module_wrapper_8_conv2d_2_kernel_momentum_read_readvariableopFsavev2_sgd_module_wrapper_8_conv2d_2_bias_momentum_read_readvariableopUsavev2_sgd_module_wrapper_10_batch_normalization_2_gamma_momentum_read_readvariableopTsavev2_sgd_module_wrapper_10_batch_normalization_2_beta_momentum_read_readvariableopIsavev2_sgd_module_wrapper_12_conv2d_3_kernel_momentum_read_readvariableopGsavev2_sgd_module_wrapper_12_conv2d_3_bias_momentum_read_readvariableopUsavev2_sgd_module_wrapper_14_batch_normalization_3_gamma_momentum_read_readvariableopTsavev2_sgd_module_wrapper_14_batch_normalization_3_beta_momentum_read_readvariableopFsavev2_sgd_module_wrapper_18_dense_kernel_momentum_read_readvariableopDsavev2_sgd_module_wrapper_18_dense_bias_momentum_read_readvariableopUsavev2_sgd_module_wrapper_20_batch_normalization_4_gamma_momentum_read_readvariableopTsavev2_sgd_module_wrapper_20_batch_normalization_4_beta_momentum_read_readvariableopHsavev2_sgd_module_wrapper_22_dense_1_kernel_momentum_read_readvariableopFsavev2_sgd_module_wrapper_22_dense_1_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *M
dtypesC
A2?	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ш
_input_shapesс
Я: : : : : : : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:@ђ:ђ:ђ:ђ:ђ:ђ:
ђђ:ђ:ђ:ђ:ђ:ђ:	ђ:: : : : : : : : : @:@:@:@:@@:@:@:@:@ђ:ђ:ђ:ђ:
ђђ:ђ:ђ:ђ:	ђ:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:! 

_output_shapes	
:ђ:!!

_output_shapes	
:ђ:!"

_output_shapes	
:ђ:%#!

_output_shapes
:	ђ: $

_output_shapes
::%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :,)(
&
_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: : ,

_output_shapes
: :,-(
&
_output_shapes
: @: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@:,1(
&
_output_shapes
:@@: 2

_output_shapes
:@: 3

_output_shapes
:@: 4

_output_shapes
:@:-5)
'
_output_shapes
:@ђ:!6

_output_shapes	
:ђ:!7

_output_shapes	
:ђ:!8

_output_shapes	
:ђ:&9"
 
_output_shapes
:
ђђ:!:

_output_shapes	
:ђ:!;

_output_shapes	
:ђ:!<

_output_shapes	
:ђ:%=!

_output_shapes
:	ђ: >

_output_shapes
::?

_output_shapes
: 
Ж
Џ
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_59879

args_0<
-batch_normalization_3_readvariableop_resource:	ђ>
/batch_normalization_3_readvariableop_1_resource:	ђM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб$batch_normalization_3/AssignNewValueб&batch_normalization_3/AssignNewValue_1б5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1и
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_3/ReadVariableOpй
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_3/ReadVariableOp_1Ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1р
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_3/FusedBatchNormV3░
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue╝
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1Ў
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ђ: : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_1:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Б>
Г
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_60086

args_0L
=batch_normalization_4_assignmovingavg_readvariableop_resource:	ђN
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:	ђA
2batch_normalization_4_cast_readvariableop_resource:	ђC
4batch_normalization_4_cast_1_readvariableop_resource:	ђ
identityѕб%batch_normalization_4/AssignMovingAvgб4batch_normalization_4/AssignMovingAvg/ReadVariableOpб'batch_normalization_4/AssignMovingAvg_1б6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpб)batch_normalization_4/Cast/ReadVariableOpб+batch_normalization_4/Cast_1/ReadVariableOpХ
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesм
"batch_normalization_4/moments/meanMeanargs_0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2$
"batch_normalization_4/moments/mean┐
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	ђ2,
*batch_normalization_4/moments/StopGradientу
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceargs_03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ21
/batch_normalization_4/moments/SquaredDifferenceЙ
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indicesІ
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2(
&batch_normalization_4/moments/variance├
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze╦
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1Ъ
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_4/AssignMovingAvg/decayу
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpы
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_4/AssignMovingAvg/subУ
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_4/AssignMovingAvg/mulГ
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_4/AssignMovingAvgБ
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_4/AssignMovingAvg_1/decayь
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpщ
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_4/AssignMovingAvg_1/sub­
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_4/AssignMovingAvg_1/mulи
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_4/AssignMovingAvg_1к
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp╠
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOpЊ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_4/batchnorm/add/y█
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/addд
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/RsqrtО
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/mul╣
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/mul_1н
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/mul_2Н
#batch_normalization_4/batchnorm/subSub1batch_normalization_4/Cast/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/subя
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/add_1џ
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
і
Џ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_58393

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ц
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_57642

args_0
identityl
activation/ReluReluargs_0*
T0*/
_output_shapes
:         dd 2
activation/Reluy
IdentityIdentityactivation/Relu:activations:0*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
Ј
h
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_60022

args_0
identityi
activation_4/ReluReluargs_0*
T0*(
_output_shapes
:         ђ2
activation_4/Relut
IdentityIdentityactivation_4/Relu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
е
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_58365

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
З

ў
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_56895

args_08
$dense_matmul_readvariableop_resource:
ђђ4
%dense_biasadd_readvariableop_resource:	ђ
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense/MatMul/ReadVariableOpє
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/BiasAddе
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ь
g
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_57475

args_0
identity▓
max_pooling2d_1/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool|
IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
џ
К
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_59721

args_0;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1Х
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1ж
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1╬
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3╚
IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : : : 2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_1:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
З

ў
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_57203

args_08
$dense_matmul_readvariableop_resource:
ђђ4
%dense_biasadd_readvariableop_resource:	ђ
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense/MatMul/ReadVariableOpє
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/BiasAddе
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
ц
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_59408

args_0
identityl
activation/ReluReluargs_0*
T0*/
_output_shapes
:         dd 2
activation/Reluy
IdentityIdentityactivation/Relu:activations:0*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
Ќ
╠
1__inference_module_wrapper_10_layer_call_fn_59752

args_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_567952
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
№
н
5__inference_batch_normalization_3_layer_call_fn_60432

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_586692
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Њ
h
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_56966

args_0
identityq
activation_5/SoftmaxSoftmaxargs_0*
T0*'
_output_shapes
:         2
activation_5/Softmaxr
IdentityIdentityactivation_5/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameargs_0
р
╬
3__inference_batch_normalization_layer_call_fn_60259

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_582992
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ж
L
0__inference_module_wrapper_3_layer_call_fn_59500

args_0
identityЛ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_566942
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
Ў
к
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_56737

args_0;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1Х
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1ж
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1╬
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !!@:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3╚
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         !!@: : : : 2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_1:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
Є
н
5__inference_batch_normalization_4_layer_call_fn_60512

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588092
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ј
е
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_57561

args_0A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@
identityѕбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOpЙ
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpг
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2
conv2d_1/BiasAddИ
IdentityIdentityconv2d_1/BiasAdd:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         !! : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         !! 
 
_user_specified_nameargs_0
З

ў
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_59994

args_08
$dense_matmul_readvariableop_resource:
ђђ4
%dense_biasadd_readvariableop_resource:	ђ
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense/MatMul/ReadVariableOpє
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/BiasAddе
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ю
ъ
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_60159

args_09
&dense_1_matmul_readvariableop_resource:	ђ5
'dense_1_biasadd_readvariableop_resource:
identityѕбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
dense_1/MatMul/ReadVariableOpІ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddГ
IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
у
л
5__inference_batch_normalization_1_layer_call_fn_60308

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_583932
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
П
ъ
I__inference_module_wrapper_layer_call_and_return_conditional_losses_59385

args_0?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpИ
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd 2
conv2d/BiasAdd▓
IdentityIdentityconv2d/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameargs_0
У
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_57581

args_0
identity«
max_pooling2d/MaxPoolMaxPoolargs_0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolz
IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
Ъ
л
1__inference_module_wrapper_14_layer_call_fn_59892

args_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_568532
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
»
h
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_56833

args_0
identityq
activation_3/ReluReluargs_0*
T0*0
_output_shapes
:         ђ2
activation_3/Relu|
IdentityIdentityactivation_3/Relu:activations:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ј
е
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_59525

args_0A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@
identityѕбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOpЙ
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpг
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2
conv2d_1/BiasAddИ
IdentityIdentityconv2d_1/BiasAdd:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         !! : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         !! 
 
_user_specified_nameargs_0
В
M
1__inference_module_wrapper_11_layer_call_fn_59785

args_0
identityм
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_573692
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
з
h
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_59910

args_0
identity│
max_pooling2d_3/MaxPoolMaxPoolargs_0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool}
IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
л
M
1__inference_module_wrapper_21_layer_call_fn_60134

args_0
identity╦
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_569432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Л
Ц
0__inference_module_wrapper_4_layer_call_fn_59543

args_0!
unknown: @
	unknown_0:@
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_575612
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         !! : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameargs_0
б)
Н
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_60499

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ+
cast_readvariableop_resource:	ђ-
cast_1_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЦ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpЎ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/subљ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpА
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/subў
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1ё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1ђ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
«
k
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_60129

args_0
identityѕw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Constњ
dropout_1/dropout/MulMulargs_0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_1/dropout/Mulh
dropout_1/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeМ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЅ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yу
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2 
dropout_1/dropout/GreaterEqualъ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_1/dropout/CastБ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_1/dropout/Mul_1p
IdentityIdentitydropout_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
с
╬
3__inference_batch_normalization_layer_call_fn_60246

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_582552
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
џ
Ъ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_58669

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ф
h
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_56875

args_0
identitys
dropout/IdentityIdentityargs_0*
T0*0
_output_shapes
:         ђ2
dropout/Identityv
IdentityIdentitydropout/Identity:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
У
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_59495

args_0
identity«
max_pooling2d/MaxPoolMaxPoolargs_0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolz
IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
кќ
л0
!__inference__traced_restore_60930
file_prefix#
assignvariableop_sgd_iter:	 &
assignvariableop_1_sgd_decay: .
$assignvariableop_2_sgd_learning_rate: )
assignvariableop_3_sgd_momentum: I
/assignvariableop_4_module_wrapper_conv2d_kernel: ;
-assignvariableop_5_module_wrapper_conv2d_bias: K
=assignvariableop_6_module_wrapper_2_batch_normalization_gamma: J
<assignvariableop_7_module_wrapper_2_batch_normalization_beta: Q
Cassignvariableop_8_module_wrapper_2_batch_normalization_moving_mean: U
Gassignvariableop_9_module_wrapper_2_batch_normalization_moving_variance: N
4assignvariableop_10_module_wrapper_4_conv2d_1_kernel: @@
2assignvariableop_11_module_wrapper_4_conv2d_1_bias:@N
@assignvariableop_12_module_wrapper_6_batch_normalization_1_gamma:@M
?assignvariableop_13_module_wrapper_6_batch_normalization_1_beta:@T
Fassignvariableop_14_module_wrapper_6_batch_normalization_1_moving_mean:@X
Jassignvariableop_15_module_wrapper_6_batch_normalization_1_moving_variance:@N
4assignvariableop_16_module_wrapper_8_conv2d_2_kernel:@@@
2assignvariableop_17_module_wrapper_8_conv2d_2_bias:@O
Aassignvariableop_18_module_wrapper_10_batch_normalization_2_gamma:@N
@assignvariableop_19_module_wrapper_10_batch_normalization_2_beta:@U
Gassignvariableop_20_module_wrapper_10_batch_normalization_2_moving_mean:@Y
Kassignvariableop_21_module_wrapper_10_batch_normalization_2_moving_variance:@P
5assignvariableop_22_module_wrapper_12_conv2d_3_kernel:@ђB
3assignvariableop_23_module_wrapper_12_conv2d_3_bias:	ђP
Aassignvariableop_24_module_wrapper_14_batch_normalization_3_gamma:	ђO
@assignvariableop_25_module_wrapper_14_batch_normalization_3_beta:	ђV
Gassignvariableop_26_module_wrapper_14_batch_normalization_3_moving_mean:	ђZ
Kassignvariableop_27_module_wrapper_14_batch_normalization_3_moving_variance:	ђF
2assignvariableop_28_module_wrapper_18_dense_kernel:
ђђ?
0assignvariableop_29_module_wrapper_18_dense_bias:	ђP
Aassignvariableop_30_module_wrapper_20_batch_normalization_4_gamma:	ђO
@assignvariableop_31_module_wrapper_20_batch_normalization_4_beta:	ђV
Gassignvariableop_32_module_wrapper_20_batch_normalization_4_moving_mean:	ђZ
Kassignvariableop_33_module_wrapper_20_batch_normalization_4_moving_variance:	ђG
4assignvariableop_34_module_wrapper_22_dense_1_kernel:	ђ@
2assignvariableop_35_module_wrapper_22_dense_1_bias:#
assignvariableop_36_total: #
assignvariableop_37_count: %
assignvariableop_38_total_1: %
assignvariableop_39_count_1: W
=assignvariableop_40_sgd_module_wrapper_conv2d_kernel_momentum: I
;assignvariableop_41_sgd_module_wrapper_conv2d_bias_momentum: Y
Kassignvariableop_42_sgd_module_wrapper_2_batch_normalization_gamma_momentum: X
Jassignvariableop_43_sgd_module_wrapper_2_batch_normalization_beta_momentum: [
Aassignvariableop_44_sgd_module_wrapper_4_conv2d_1_kernel_momentum: @M
?assignvariableop_45_sgd_module_wrapper_4_conv2d_1_bias_momentum:@[
Massignvariableop_46_sgd_module_wrapper_6_batch_normalization_1_gamma_momentum:@Z
Lassignvariableop_47_sgd_module_wrapper_6_batch_normalization_1_beta_momentum:@[
Aassignvariableop_48_sgd_module_wrapper_8_conv2d_2_kernel_momentum:@@M
?assignvariableop_49_sgd_module_wrapper_8_conv2d_2_bias_momentum:@\
Nassignvariableop_50_sgd_module_wrapper_10_batch_normalization_2_gamma_momentum:@[
Massignvariableop_51_sgd_module_wrapper_10_batch_normalization_2_beta_momentum:@]
Bassignvariableop_52_sgd_module_wrapper_12_conv2d_3_kernel_momentum:@ђO
@assignvariableop_53_sgd_module_wrapper_12_conv2d_3_bias_momentum:	ђ]
Nassignvariableop_54_sgd_module_wrapper_14_batch_normalization_3_gamma_momentum:	ђ\
Massignvariableop_55_sgd_module_wrapper_14_batch_normalization_3_beta_momentum:	ђS
?assignvariableop_56_sgd_module_wrapper_18_dense_kernel_momentum:
ђђL
=assignvariableop_57_sgd_module_wrapper_18_dense_bias_momentum:	ђ]
Nassignvariableop_58_sgd_module_wrapper_20_batch_normalization_4_gamma_momentum:	ђ\
Massignvariableop_59_sgd_module_wrapper_20_batch_normalization_4_beta_momentum:	ђT
Aassignvariableop_60_sgd_module_wrapper_22_dense_1_kernel_momentum:	ђM
?assignvariableop_61_sgd_module_wrapper_22_dense_1_bias_momentum:
identity_63ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9¤
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*█
valueЛB╬?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/30/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/31/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*Њ
valueЅBє?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesж
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*њ
_output_shapes 
Ч:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identityў
AssignVariableOpAssignVariableOpassignvariableop_sgd_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1А
AssignVariableOp_1AssignVariableOpassignvariableop_1_sgd_decayIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Е
AssignVariableOp_2AssignVariableOp$assignvariableop_2_sgd_learning_rateIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ц
AssignVariableOp_3AssignVariableOpassignvariableop_3_sgd_momentumIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4┤
AssignVariableOp_4AssignVariableOp/assignvariableop_4_module_wrapper_conv2d_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5▓
AssignVariableOp_5AssignVariableOp-assignvariableop_5_module_wrapper_conv2d_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6┬
AssignVariableOp_6AssignVariableOp=assignvariableop_6_module_wrapper_2_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7┴
AssignVariableOp_7AssignVariableOp<assignvariableop_7_module_wrapper_2_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╚
AssignVariableOp_8AssignVariableOpCassignvariableop_8_module_wrapper_2_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╠
AssignVariableOp_9AssignVariableOpGassignvariableop_9_module_wrapper_2_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╝
AssignVariableOp_10AssignVariableOp4assignvariableop_10_module_wrapper_4_conv2d_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11║
AssignVariableOp_11AssignVariableOp2assignvariableop_11_module_wrapper_4_conv2d_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╚
AssignVariableOp_12AssignVariableOp@assignvariableop_12_module_wrapper_6_batch_normalization_1_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13К
AssignVariableOp_13AssignVariableOp?assignvariableop_13_module_wrapper_6_batch_normalization_1_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14╬
AssignVariableOp_14AssignVariableOpFassignvariableop_14_module_wrapper_6_batch_normalization_1_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15м
AssignVariableOp_15AssignVariableOpJassignvariableop_15_module_wrapper_6_batch_normalization_1_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╝
AssignVariableOp_16AssignVariableOp4assignvariableop_16_module_wrapper_8_conv2d_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17║
AssignVariableOp_17AssignVariableOp2assignvariableop_17_module_wrapper_8_conv2d_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╔
AssignVariableOp_18AssignVariableOpAassignvariableop_18_module_wrapper_10_batch_normalization_2_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╚
AssignVariableOp_19AssignVariableOp@assignvariableop_19_module_wrapper_10_batch_normalization_2_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¤
AssignVariableOp_20AssignVariableOpGassignvariableop_20_module_wrapper_10_batch_normalization_2_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21М
AssignVariableOp_21AssignVariableOpKassignvariableop_21_module_wrapper_10_batch_normalization_2_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22й
AssignVariableOp_22AssignVariableOp5assignvariableop_22_module_wrapper_12_conv2d_3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╗
AssignVariableOp_23AssignVariableOp3assignvariableop_23_module_wrapper_12_conv2d_3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╔
AssignVariableOp_24AssignVariableOpAassignvariableop_24_module_wrapper_14_batch_normalization_3_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╚
AssignVariableOp_25AssignVariableOp@assignvariableop_25_module_wrapper_14_batch_normalization_3_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¤
AssignVariableOp_26AssignVariableOpGassignvariableop_26_module_wrapper_14_batch_normalization_3_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27М
AssignVariableOp_27AssignVariableOpKassignvariableop_27_module_wrapper_14_batch_normalization_3_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28║
AssignVariableOp_28AssignVariableOp2assignvariableop_28_module_wrapper_18_dense_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29И
AssignVariableOp_29AssignVariableOp0assignvariableop_29_module_wrapper_18_dense_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╔
AssignVariableOp_30AssignVariableOpAassignvariableop_30_module_wrapper_20_batch_normalization_4_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╚
AssignVariableOp_31AssignVariableOp@assignvariableop_31_module_wrapper_20_batch_normalization_4_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¤
AssignVariableOp_32AssignVariableOpGassignvariableop_32_module_wrapper_20_batch_normalization_4_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33М
AssignVariableOp_33AssignVariableOpKassignvariableop_33_module_wrapper_20_batch_normalization_4_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╝
AssignVariableOp_34AssignVariableOp4assignvariableop_34_module_wrapper_22_dense_1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35║
AssignVariableOp_35AssignVariableOp2assignvariableop_35_module_wrapper_22_dense_1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36А
AssignVariableOp_36AssignVariableOpassignvariableop_36_totalIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37А
AssignVariableOp_37AssignVariableOpassignvariableop_37_countIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Б
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Б
AssignVariableOp_39AssignVariableOpassignvariableop_39_count_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40┼
AssignVariableOp_40AssignVariableOp=assignvariableop_40_sgd_module_wrapper_conv2d_kernel_momentumIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41├
AssignVariableOp_41AssignVariableOp;assignvariableop_41_sgd_module_wrapper_conv2d_bias_momentumIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42М
AssignVariableOp_42AssignVariableOpKassignvariableop_42_sgd_module_wrapper_2_batch_normalization_gamma_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43м
AssignVariableOp_43AssignVariableOpJassignvariableop_43_sgd_module_wrapper_2_batch_normalization_beta_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44╔
AssignVariableOp_44AssignVariableOpAassignvariableop_44_sgd_module_wrapper_4_conv2d_1_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45К
AssignVariableOp_45AssignVariableOp?assignvariableop_45_sgd_module_wrapper_4_conv2d_1_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Н
AssignVariableOp_46AssignVariableOpMassignvariableop_46_sgd_module_wrapper_6_batch_normalization_1_gamma_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47н
AssignVariableOp_47AssignVariableOpLassignvariableop_47_sgd_module_wrapper_6_batch_normalization_1_beta_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48╔
AssignVariableOp_48AssignVariableOpAassignvariableop_48_sgd_module_wrapper_8_conv2d_2_kernel_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49К
AssignVariableOp_49AssignVariableOp?assignvariableop_49_sgd_module_wrapper_8_conv2d_2_bias_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50о
AssignVariableOp_50AssignVariableOpNassignvariableop_50_sgd_module_wrapper_10_batch_normalization_2_gamma_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Н
AssignVariableOp_51AssignVariableOpMassignvariableop_51_sgd_module_wrapper_10_batch_normalization_2_beta_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52╩
AssignVariableOp_52AssignVariableOpBassignvariableop_52_sgd_module_wrapper_12_conv2d_3_kernel_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53╚
AssignVariableOp_53AssignVariableOp@assignvariableop_53_sgd_module_wrapper_12_conv2d_3_bias_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54о
AssignVariableOp_54AssignVariableOpNassignvariableop_54_sgd_module_wrapper_14_batch_normalization_3_gamma_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Н
AssignVariableOp_55AssignVariableOpMassignvariableop_55_sgd_module_wrapper_14_batch_normalization_3_beta_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56К
AssignVariableOp_56AssignVariableOp?assignvariableop_56_sgd_module_wrapper_18_dense_kernel_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57┼
AssignVariableOp_57AssignVariableOp=assignvariableop_57_sgd_module_wrapper_18_dense_bias_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58о
AssignVariableOp_58AssignVariableOpNassignvariableop_58_sgd_module_wrapper_20_batch_normalization_4_gamma_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Н
AssignVariableOp_59AssignVariableOpMassignvariableop_59_sgd_module_wrapper_20_batch_normalization_4_beta_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60╔
AssignVariableOp_60AssignVariableOpAassignvariableop_60_sgd_module_wrapper_22_dense_1_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61К
AssignVariableOp_61AssignVariableOp?assignvariableop_61_sgd_module_wrapper_22_dense_1_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_619
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp▓
Identity_62Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_62Ц
Identity_63IdentityIdentity_62:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_63"#
identity_63Identity_63:output:0*њ
_input_shapesђ
~: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ж
Џ
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_57300

args_0<
-batch_normalization_3_readvariableop_resource:	ђ>
/batch_normalization_3_readvariableop_1_resource:	ђM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб$batch_normalization_3/AssignNewValueб&batch_normalization_3/AssignNewValue_1б5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1и
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_3/ReadVariableOpй
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_3/ReadVariableOp_1Ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1р
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_3/FusedBatchNormV3░
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue╝
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1Ў
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ђ: : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_1:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
▓
А
1__inference_module_wrapper_18_layer_call_fn_60003

args_0
unknown:
ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_568952
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Њ
h
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_60187

args_0
identityq
activation_5/SoftmaxSoftmaxargs_0*
T0*'
_output_shapes
:         2
activation_5/Softmaxr
IdentityIdentityactivation_5/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameargs_0
і
Џ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_60339

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
▄
j
1__inference_module_wrapper_21_layer_call_fn_60139

args_0
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_571012
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
╝
й
N__inference_batch_normalization_layer_call_and_return_conditional_losses_58299

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
б)
Н
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_58869

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ+
cast_readvariableop_resource:	ђ-
cast_1_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЦ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpЎ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/subљ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpА
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/subў
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1ё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1ђ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╦
Х
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_56679

args_09
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб3batch_normalization/FusedBatchNormV3/ReadVariableOpб5batch_normalization/FusedBatchNormV3/ReadVariableOp_1б"batch_normalization/ReadVariableOpб$batch_normalization/ReadVariableOp_1░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpХ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1с
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpж
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1┬
$batch_normalization/FusedBatchNormV3FusedBatchNormV3args_0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oЃ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3Й
IdentityIdentity(batch_normalization/FusedBatchNormV3:y:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         dd : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_1:W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
Л
Ц
0__inference_module_wrapper_8_layer_call_fn_59683

args_0!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_574552
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
ф
g
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_59548

args_0
identityp
activation_1/ReluReluargs_0*
T0*/
_output_shapes
:         !!@2
activation_1/Relu{
IdentityIdentityactivation_1/Relu:activations:0*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
К
с
#__inference_signature_wrapper_58232
module_wrapper_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@ђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ

unknown_21:	ђ

unknown_22:	ђ

unknown_23:
ђђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ

unknown_27:	ђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__wrapped_model_566312
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:         dd
.
_user_specified_namemodule_wrapper_input
┐
▄
*__inference_sequential_layer_call_fn_59365

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@ђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ

unknown_21:	ђ

unknown_22:	ђ

unknown_23:
ђђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ

unknown_27:	ђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:
identityѕбStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_578372
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
д
│
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_60052

args_0A
2batch_normalization_4_cast_readvariableop_resource:	ђC
4batch_normalization_4_cast_1_readvariableop_resource:	ђC
4batch_normalization_4_cast_2_readvariableop_resource:	ђC
4batch_normalization_4_cast_3_readvariableop_resource:	ђ
identityѕб)batch_normalization_4/Cast/ReadVariableOpб+batch_normalization_4/Cast_1/ReadVariableOpб+batch_normalization_4/Cast_2/ReadVariableOpб+batch_normalization_4/Cast_3/ReadVariableOpк
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp╠
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOp╠
+batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_2/ReadVariableOp╠
+batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_3/ReadVariableOpЊ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_4/batchnorm/add/yя
#batch_normalization_4/batchnorm/addAddV23batch_normalization_4/Cast_1/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/addд
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/RsqrtО
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/mul╣
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/mul_1О
%batch_normalization_4/batchnorm/mul_2Mul1batch_normalization_4/Cast/ReadVariableOp:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/mul_2О
#batch_normalization_4/batchnorm/subSub3batch_normalization_4/Cast_2/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/subя
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/add_1┤
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp,^batch_normalization_4/Cast_2/ReadVariableOp,^batch_normalization_4/Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2Z
+batch_normalization_4/Cast_2/ReadVariableOp+batch_normalization_4/Cast_2/ReadVariableOp2Z
+batch_normalization_4/Cast_3/ReadVariableOp+batch_normalization_4/Cast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
­
M
1__inference_module_wrapper_13_layer_call_fn_59838

args_0
identityМ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_568332
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Й
k
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_59942

args_0
identityѕs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/dropout/Constћ
dropout/dropout/MulMulargs_0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout/dropout/Muld
dropout/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout/dropout/ShapeН
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2 
dropout/dropout/GreaterEqual/yу
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2
dropout/dropout/GreaterEqualа
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout/dropout/CastБ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout/dropout/Mul_1v
IdentityIdentitydropout/dropout/Mul_1:z:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ў
к
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_59581

args_0;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1Х
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1ж
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1╬
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !!@:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3╚
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         !!@: : : : 2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_1:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
 
л
1__inference_module_wrapper_20_layer_call_fn_60099

args_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_569282
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ё
н
5__inference_batch_normalization_4_layer_call_fn_60525

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588692
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
т
л
5__inference_batch_normalization_2_layer_call_fn_60383

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_585752
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ќ
Ф
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_59795

args_0B
'conv2d_3_conv2d_readvariableop_resource:@ђ7
(conv2d_3_biasadd_readvariableop_resource:	ђ
identityѕбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOp▒
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_3/Conv2D/ReadVariableOp┐
conv2d_3/Conv2DConv2Dargs_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_3/Conv2Dе
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpГ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_3/BiasAdd╣
IdentityIdentityconv2d_3/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
№
h
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_56810

args_0
identity▓
max_pooling2d_2/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool|
IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
П
ъ
I__inference_module_wrapper_layer_call_and_return_conditional_losses_57667

args_0?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpИ
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd 2
conv2d/BiasAdd▓
IdentityIdentityconv2d/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameargs_0
з
Ж
*__inference_sequential_layer_call_fn_57036
module_wrapper_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@ђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ

unknown_21:	ђ

unknown_22:	ђ

unknown_23:
ђђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ

unknown_27:	ђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_569692
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:         dd
.
_user_specified_namemodule_wrapper_input
ф
g
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_59693

args_0
identityp
activation_2/ReluReluargs_0*
T0*/
_output_shapes
:         @2
activation_2/Relu{
IdentityIdentityactivation_2/Relu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ю
ъ
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_57074

args_09
&dense_1_matmul_readvariableop_resource:	ђ5
'dense_1_biasadd_readvariableop_resource:
identityѕбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
dense_1/MatMul/ReadVariableOpІ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddГ
IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
╦
Х
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_59441

args_09
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб3batch_normalization/FusedBatchNormV3/ReadVariableOpб5batch_normalization/FusedBatchNormV3/ReadVariableOp_1б"batch_normalization/ReadVariableOpб$batch_normalization/ReadVariableOp_1░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpХ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1с
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpж
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1┬
$batch_normalization/FusedBatchNormV3FusedBatchNormV3args_0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oЃ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3Й
IdentityIdentity(batch_normalization/FusedBatchNormV3:y:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         dd : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_1:W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
л
M
1__inference_module_wrapper_19_layer_call_fn_60032

args_0
identity╦
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_571782
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
ц
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_56659

args_0
identityl
activation/ReluReluargs_0*
T0*/
_output_shapes
:         dd 2
activation/Reluy
IdentityIdentityactivation/Relu:activations:0*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
з
h
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_56868

args_0
identity│
max_pooling2d_3/MaxPoolMaxPoolargs_0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool}
IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ћ
╦
0__inference_module_wrapper_2_layer_call_fn_59472

args_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_566792
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         dd : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
д
│
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_56928

args_0A
2batch_normalization_4_cast_readvariableop_resource:	ђC
4batch_normalization_4_cast_1_readvariableop_resource:	ђC
4batch_normalization_4_cast_2_readvariableop_resource:	ђC
4batch_normalization_4_cast_3_readvariableop_resource:	ђ
identityѕб)batch_normalization_4/Cast/ReadVariableOpб+batch_normalization_4/Cast_1/ReadVariableOpб+batch_normalization_4/Cast_2/ReadVariableOpб+batch_normalization_4/Cast_3/ReadVariableOpк
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp╠
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOp╠
+batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_2/ReadVariableOp╠
+batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_3/ReadVariableOpЊ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_4/batchnorm/add/yя
#batch_normalization_4/batchnorm/addAddV23batch_normalization_4/Cast_1/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/addд
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/RsqrtО
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/mul╣
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/mul_1О
%batch_normalization_4/batchnorm/mul_2Mul1batch_normalization_4/Cast/ReadVariableOp:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/mul_2О
#batch_normalization_4/batchnorm/subSub3batch_normalization_4/Cast_2/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/subя
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/add_1┤
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp,^batch_normalization_4/Cast_2/ReadVariableOp,^batch_normalization_4/Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2Z
+batch_normalization_4/Cast_2/ReadVariableOp+batch_normalization_4/Cast_2/ReadVariableOp2Z
+batch_normalization_4/Cast_3/ReadVariableOp+batch_normalization_4/Cast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
»
h
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_57324

args_0
identityq
activation_3/ReluReluargs_0*
T0*0
_output_shapes
:         ђ2
activation_3/Relu|
IdentityIdentityactivation_3/Relu:activations:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
ф
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_58503

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
дs
с
E__inference_sequential_layer_call_and_return_conditional_losses_58065
module_wrapper_input.
module_wrapper_57976: "
module_wrapper_57978: $
module_wrapper_2_57982: $
module_wrapper_2_57984: $
module_wrapper_2_57986: $
module_wrapper_2_57988: 0
module_wrapper_4_57992: @$
module_wrapper_4_57994:@$
module_wrapper_6_57998:@$
module_wrapper_6_58000:@$
module_wrapper_6_58002:@$
module_wrapper_6_58004:@0
module_wrapper_8_58008:@@$
module_wrapper_8_58010:@%
module_wrapper_10_58014:@%
module_wrapper_10_58016:@%
module_wrapper_10_58018:@%
module_wrapper_10_58020:@2
module_wrapper_12_58024:@ђ&
module_wrapper_12_58026:	ђ&
module_wrapper_14_58030:	ђ&
module_wrapper_14_58032:	ђ&
module_wrapper_14_58034:	ђ&
module_wrapper_14_58036:	ђ+
module_wrapper_18_58042:
ђђ&
module_wrapper_18_58044:	ђ&
module_wrapper_20_58048:	ђ&
module_wrapper_20_58050:	ђ&
module_wrapper_20_58052:	ђ&
module_wrapper_20_58054:	ђ*
module_wrapper_22_58058:	ђ%
module_wrapper_22_58060:
identityѕб&module_wrapper/StatefulPartitionedCallб)module_wrapper_10/StatefulPartitionedCallб)module_wrapper_12/StatefulPartitionedCallб)module_wrapper_14/StatefulPartitionedCallб)module_wrapper_18/StatefulPartitionedCallб(module_wrapper_2/StatefulPartitionedCallб)module_wrapper_20/StatefulPartitionedCallб)module_wrapper_22/StatefulPartitionedCallб(module_wrapper_4/StatefulPartitionedCallб(module_wrapper_6/StatefulPartitionedCallб(module_wrapper_8/StatefulPartitionedCall┼
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_57976module_wrapper_57978*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_566482(
&module_wrapper/StatefulPartitionedCallю
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_566592"
 module_wrapper_1/PartitionedCallў
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0module_wrapper_2_57982module_wrapper_2_57984module_wrapper_2_57986module_wrapper_2_57988*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_566792*
(module_wrapper_2/StatefulPartitionedCallъ
 module_wrapper_3/PartitionedCallPartitionedCall1module_wrapper_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_566942"
 module_wrapper_3/PartitionedCallС
(module_wrapper_4/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0module_wrapper_4_57992module_wrapper_4_57994*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_567062*
(module_wrapper_4/StatefulPartitionedCallъ
 module_wrapper_5/PartitionedCallPartitionedCall1module_wrapper_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_567172"
 module_wrapper_5/PartitionedCallў
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_5/PartitionedCall:output:0module_wrapper_6_57998module_wrapper_6_58000module_wrapper_6_58002module_wrapper_6_58004*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_567372*
(module_wrapper_6/StatefulPartitionedCallъ
 module_wrapper_7/PartitionedCallPartitionedCall1module_wrapper_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_567522"
 module_wrapper_7/PartitionedCallС
(module_wrapper_8/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_7/PartitionedCall:output:0module_wrapper_8_58008module_wrapper_8_58010*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_567642*
(module_wrapper_8/StatefulPartitionedCallъ
 module_wrapper_9/PartitionedCallPartitionedCall1module_wrapper_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_567752"
 module_wrapper_9/PartitionedCallЪ
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_58014module_wrapper_10_58016module_wrapper_10_58018module_wrapper_10_58020*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_567952+
)module_wrapper_10/StatefulPartitionedCallб
!module_wrapper_11/PartitionedCallPartitionedCall2module_wrapper_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_568102#
!module_wrapper_11/PartitionedCallв
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_11/PartitionedCall:output:0module_wrapper_12_58024module_wrapper_12_58026*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_568222+
)module_wrapper_12/StatefulPartitionedCallБ
!module_wrapper_13/PartitionedCallPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_568332#
!module_wrapper_13/PartitionedCallА
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_13/PartitionedCall:output:0module_wrapper_14_58030module_wrapper_14_58032module_wrapper_14_58034module_wrapper_14_58036*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_568532+
)module_wrapper_14/StatefulPartitionedCallБ
!module_wrapper_15/PartitionedCallPartitionedCall2module_wrapper_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_568682#
!module_wrapper_15/PartitionedCallЏ
!module_wrapper_16/PartitionedCallPartitionedCall*module_wrapper_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_568752#
!module_wrapper_16/PartitionedCallЊ
!module_wrapper_17/PartitionedCallPartitionedCall*module_wrapper_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_568832#
!module_wrapper_17/PartitionedCallс
)module_wrapper_18/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_17/PartitionedCall:output:0module_wrapper_18_58042module_wrapper_18_58044*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_568952+
)module_wrapper_18/StatefulPartitionedCallЏ
!module_wrapper_19/PartitionedCallPartitionedCall2module_wrapper_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_569062#
!module_wrapper_19/PartitionedCallЎ
)module_wrapper_20/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_19/PartitionedCall:output:0module_wrapper_20_58048module_wrapper_20_58050module_wrapper_20_58052module_wrapper_20_58054*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_569282+
)module_wrapper_20/StatefulPartitionedCallЏ
!module_wrapper_21/PartitionedCallPartitionedCall2module_wrapper_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_569432#
!module_wrapper_21/PartitionedCallР
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_21/PartitionedCall:output:0module_wrapper_22_58058module_wrapper_22_58060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_569552+
)module_wrapper_22/StatefulPartitionedCallџ
!module_wrapper_23/PartitionedCallPartitionedCall2module_wrapper_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_569662#
!module_wrapper_23/PartitionedCall█
IdentityIdentity*module_wrapper_23/PartitionedCall:output:0'^module_wrapper/StatefulPartitionedCall*^module_wrapper_10/StatefulPartitionedCall*^module_wrapper_12/StatefulPartitionedCall*^module_wrapper_14/StatefulPartitionedCall*^module_wrapper_18/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall*^module_wrapper_20/StatefulPartitionedCall*^module_wrapper_22/StatefulPartitionedCall)^module_wrapper_4/StatefulPartitionedCall)^module_wrapper_6/StatefulPartitionedCall)^module_wrapper_8/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2V
)module_wrapper_10/StatefulPartitionedCall)module_wrapper_10/StatefulPartitionedCall2V
)module_wrapper_12/StatefulPartitionedCall)module_wrapper_12/StatefulPartitionedCall2V
)module_wrapper_14/StatefulPartitionedCall)module_wrapper_14/StatefulPartitionedCall2V
)module_wrapper_18/StatefulPartitionedCall)module_wrapper_18/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall2V
)module_wrapper_20/StatefulPartitionedCall)module_wrapper_20/StatefulPartitionedCall2V
)module_wrapper_22/StatefulPartitionedCall)module_wrapper_22/StatefulPartitionedCall2T
(module_wrapper_4/StatefulPartitionedCall(module_wrapper_4/StatefulPartitionedCall2T
(module_wrapper_6/StatefulPartitionedCall(module_wrapper_6/StatefulPartitionedCall2T
(module_wrapper_8/StatefulPartitionedCall(module_wrapper_8/StatefulPartitionedCall:e a
/
_output_shapes
:         dd
.
_user_specified_namemodule_wrapper_input
Љ
h
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_56943

args_0
identityo
dropout_1/IdentityIdentityargs_0*
T0*(
_output_shapes
:         ђ2
dropout_1/Identityp
IdentityIdentitydropout_1/Identity:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Њ
╦
0__inference_module_wrapper_6_layer_call_fn_59625

args_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_575122
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         !!@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
Ю
ъ
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_56955

args_09
&dense_1_matmul_readvariableop_resource:	ђ5
'dense_1_biasadd_readvariableop_resource:
identityѕбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
dense_1/MatMul/ReadVariableOpІ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddГ
IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ј
е
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_59655

args_0A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@
identityѕбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOp░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpЙ
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_2/Conv2DД
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpг
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAddИ
IdentityIdentityconv2d_2/BiasAdd:output:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ь
g
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_56752

args_0
identity▓
max_pooling2d_1/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool|
IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
Л
Ц
0__inference_module_wrapper_8_layer_call_fn_59674

args_0!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_567642
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ј
е
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_56764

args_0A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@
identityѕбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOp░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpЙ
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_2/Conv2DД
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpг
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAddИ
IdentityIdentityconv2d_2/BiasAdd:output:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
«
k
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_57101

args_0
identityѕw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Constњ
dropout_1/dropout/MulMulargs_0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_1/dropout/Mulh
dropout_1/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeМ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЅ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yу
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2 
dropout_1/dropout/GreaterEqualъ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_1/dropout/CastБ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_1/dropout/Mul_1p
IdentityIdentitydropout_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
А
h
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_59964

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/Constђ
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
«
Ъ
1__inference_module_wrapper_22_layer_call_fn_60177

args_0
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_570742
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ј
h
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_60017

args_0
identityi
activation_4/ReluReluargs_0*
T0*(
_output_shapes
:         ђ2
activation_4/Relut
IdentityIdentityactivation_4/Relu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Уv
╗
E__inference_sequential_layer_call_and_return_conditional_losses_58157
module_wrapper_input.
module_wrapper_58068: "
module_wrapper_58070: $
module_wrapper_2_58074: $
module_wrapper_2_58076: $
module_wrapper_2_58078: $
module_wrapper_2_58080: 0
module_wrapper_4_58084: @$
module_wrapper_4_58086:@$
module_wrapper_6_58090:@$
module_wrapper_6_58092:@$
module_wrapper_6_58094:@$
module_wrapper_6_58096:@0
module_wrapper_8_58100:@@$
module_wrapper_8_58102:@%
module_wrapper_10_58106:@%
module_wrapper_10_58108:@%
module_wrapper_10_58110:@%
module_wrapper_10_58112:@2
module_wrapper_12_58116:@ђ&
module_wrapper_12_58118:	ђ&
module_wrapper_14_58122:	ђ&
module_wrapper_14_58124:	ђ&
module_wrapper_14_58126:	ђ&
module_wrapper_14_58128:	ђ+
module_wrapper_18_58134:
ђђ&
module_wrapper_18_58136:	ђ&
module_wrapper_20_58140:	ђ&
module_wrapper_20_58142:	ђ&
module_wrapper_20_58144:	ђ&
module_wrapper_20_58146:	ђ*
module_wrapper_22_58150:	ђ%
module_wrapper_22_58152:
identityѕб&module_wrapper/StatefulPartitionedCallб)module_wrapper_10/StatefulPartitionedCallб)module_wrapper_12/StatefulPartitionedCallб)module_wrapper_14/StatefulPartitionedCallб)module_wrapper_16/StatefulPartitionedCallб)module_wrapper_18/StatefulPartitionedCallб(module_wrapper_2/StatefulPartitionedCallб)module_wrapper_20/StatefulPartitionedCallб)module_wrapper_21/StatefulPartitionedCallб)module_wrapper_22/StatefulPartitionedCallб(module_wrapper_4/StatefulPartitionedCallб(module_wrapper_6/StatefulPartitionedCallб(module_wrapper_8/StatefulPartitionedCall┼
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_58068module_wrapper_58070*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_576672(
&module_wrapper/StatefulPartitionedCallю
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_576422"
 module_wrapper_1/PartitionedCallќ
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0module_wrapper_2_58074module_wrapper_2_58076module_wrapper_2_58078module_wrapper_2_58080*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_576182*
(module_wrapper_2/StatefulPartitionedCallъ
 module_wrapper_3/PartitionedCallPartitionedCall1module_wrapper_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_575812"
 module_wrapper_3/PartitionedCallС
(module_wrapper_4/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0module_wrapper_4_58084module_wrapper_4_58086*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_575612*
(module_wrapper_4/StatefulPartitionedCallъ
 module_wrapper_5/PartitionedCallPartitionedCall1module_wrapper_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_575362"
 module_wrapper_5/PartitionedCallќ
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_5/PartitionedCall:output:0module_wrapper_6_58090module_wrapper_6_58092module_wrapper_6_58094module_wrapper_6_58096*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_575122*
(module_wrapper_6/StatefulPartitionedCallъ
 module_wrapper_7/PartitionedCallPartitionedCall1module_wrapper_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_574752"
 module_wrapper_7/PartitionedCallС
(module_wrapper_8/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_7/PartitionedCall:output:0module_wrapper_8_58100module_wrapper_8_58102*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_574552*
(module_wrapper_8/StatefulPartitionedCallъ
 module_wrapper_9/PartitionedCallPartitionedCall1module_wrapper_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_574302"
 module_wrapper_9/PartitionedCallЮ
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_58106module_wrapper_10_58108module_wrapper_10_58110module_wrapper_10_58112*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_574062+
)module_wrapper_10/StatefulPartitionedCallб
!module_wrapper_11/PartitionedCallPartitionedCall2module_wrapper_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_573692#
!module_wrapper_11/PartitionedCallв
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_11/PartitionedCall:output:0module_wrapper_12_58116module_wrapper_12_58118*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_573492+
)module_wrapper_12/StatefulPartitionedCallБ
!module_wrapper_13/PartitionedCallPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_573242#
!module_wrapper_13/PartitionedCallЪ
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_13/PartitionedCall:output:0module_wrapper_14_58122module_wrapper_14_58124module_wrapper_14_58126module_wrapper_14_58128*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_573002+
)module_wrapper_14/StatefulPartitionedCallБ
!module_wrapper_15/PartitionedCallPartitionedCall2module_wrapper_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_572632#
!module_wrapper_15/PartitionedCall│
)module_wrapper_16/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_572472+
)module_wrapper_16/StatefulPartitionedCallЏ
!module_wrapper_17/PartitionedCallPartitionedCall2module_wrapper_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_572242#
!module_wrapper_17/PartitionedCallс
)module_wrapper_18/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_17/PartitionedCall:output:0module_wrapper_18_58134module_wrapper_18_58136*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_572032+
)module_wrapper_18/StatefulPartitionedCallЏ
!module_wrapper_19/PartitionedCallPartitionedCall2module_wrapper_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_571782#
!module_wrapper_19/PartitionedCallЌ
)module_wrapper_20/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_19/PartitionedCall:output:0module_wrapper_20_58140module_wrapper_20_58142module_wrapper_20_58144module_wrapper_20_58146*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_571542+
)module_wrapper_20/StatefulPartitionedCall▀
)module_wrapper_21/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_20/StatefulPartitionedCall:output:0*^module_wrapper_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_571012+
)module_wrapper_21/StatefulPartitionedCallЖ
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_21/StatefulPartitionedCall:output:0module_wrapper_22_58150module_wrapper_22_58152*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_570742+
)module_wrapper_22/StatefulPartitionedCallџ
!module_wrapper_23/PartitionedCallPartitionedCall2module_wrapper_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_570492#
!module_wrapper_23/PartitionedCall│
IdentityIdentity*module_wrapper_23/PartitionedCall:output:0'^module_wrapper/StatefulPartitionedCall*^module_wrapper_10/StatefulPartitionedCall*^module_wrapper_12/StatefulPartitionedCall*^module_wrapper_14/StatefulPartitionedCall*^module_wrapper_16/StatefulPartitionedCall*^module_wrapper_18/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall*^module_wrapper_20/StatefulPartitionedCall*^module_wrapper_21/StatefulPartitionedCall*^module_wrapper_22/StatefulPartitionedCall)^module_wrapper_4/StatefulPartitionedCall)^module_wrapper_6/StatefulPartitionedCall)^module_wrapper_8/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2V
)module_wrapper_10/StatefulPartitionedCall)module_wrapper_10/StatefulPartitionedCall2V
)module_wrapper_12/StatefulPartitionedCall)module_wrapper_12/StatefulPartitionedCall2V
)module_wrapper_14/StatefulPartitionedCall)module_wrapper_14/StatefulPartitionedCall2V
)module_wrapper_16/StatefulPartitionedCall)module_wrapper_16/StatefulPartitionedCall2V
)module_wrapper_18/StatefulPartitionedCall)module_wrapper_18/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall2V
)module_wrapper_20/StatefulPartitionedCall)module_wrapper_20/StatefulPartitionedCall2V
)module_wrapper_21/StatefulPartitionedCall)module_wrapper_21/StatefulPartitionedCall2V
)module_wrapper_22/StatefulPartitionedCall)module_wrapper_22/StatefulPartitionedCall2T
(module_wrapper_4/StatefulPartitionedCall(module_wrapper_4/StatefulPartitionedCall2T
(module_wrapper_6/StatefulPartitionedCall(module_wrapper_6/StatefulPartitionedCall2T
(module_wrapper_8/StatefulPartitionedCall(module_wrapper_8/StatefulPartitionedCall:e a
/
_output_shapes
:         dd
.
_user_specified_namemodule_wrapper_input
№
h
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_59775

args_0
identity▓
max_pooling2d_2/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool|
IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ћ
╦
0__inference_module_wrapper_6_layer_call_fn_59612

args_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_567372
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         !!@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
џ
К
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_56795

args_0;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1Х
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1ж
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1╬
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3╚
IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : : : 2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_1:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ж
L
0__inference_module_wrapper_1_layer_call_fn_59418

args_0
identityЛ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_566592
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
Б>
Г
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_57154

args_0L
=batch_normalization_4_assignmovingavg_readvariableop_resource:	ђN
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:	ђA
2batch_normalization_4_cast_readvariableop_resource:	ђC
4batch_normalization_4_cast_1_readvariableop_resource:	ђ
identityѕб%batch_normalization_4/AssignMovingAvgб4batch_normalization_4/AssignMovingAvg/ReadVariableOpб'batch_normalization_4/AssignMovingAvg_1б6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpб)batch_normalization_4/Cast/ReadVariableOpб+batch_normalization_4/Cast_1/ReadVariableOpХ
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesм
"batch_normalization_4/moments/meanMeanargs_0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2$
"batch_normalization_4/moments/mean┐
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	ђ2,
*batch_normalization_4/moments/StopGradientу
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceargs_03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ21
/batch_normalization_4/moments/SquaredDifferenceЙ
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indicesІ
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2(
&batch_normalization_4/moments/variance├
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze╦
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1Ъ
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_4/AssignMovingAvg/decayу
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpы
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_4/AssignMovingAvg/subУ
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_4/AssignMovingAvg/mulГ
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_4/AssignMovingAvgБ
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_4/AssignMovingAvg_1/decayь
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpщ
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_4/AssignMovingAvg_1/sub­
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_4/AssignMovingAvg_1/mulи
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_4/AssignMovingAvg_1к
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp╠
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOpЊ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_4/batchnorm/add/y█
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/addд
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/RsqrtО
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/mul╣
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/mul_1н
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/mul_2Н
#batch_normalization_4/batchnorm/subSub1batch_normalization_4/Cast/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/subя
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/add_1џ
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Љ
h
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_60117

args_0
identityo
dropout_1/IdentityIdentityargs_0*
T0*(
_output_shapes
:         ђ2
dropout_1/Identityp
IdentityIdentitydropout_1/Identity:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Њ
╦
0__inference_module_wrapper_2_layer_call_fn_59485

args_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_576182
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         dd : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
­
M
1__inference_module_wrapper_15_layer_call_fn_59920

args_0
identityМ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_568682
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
ф
╦
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_56853

args_0<
-batch_normalization_3_readvariableop_resource:	ђ>
/batch_normalization_3_readvariableop_1_resource:	ђM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1и
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_3/ReadVariableOpй
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_3/ReadVariableOp_1Ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1М
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3╔
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ђ: : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_1:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Й
┐
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_58575

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ф
g
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_57536

args_0
identityp
activation_1/ReluReluargs_0*
T0*/
_output_shapes
:         !!@2
activation_1/Relu{
IdentityIdentityactivation_1/Relu:activations:0*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
ь
н
5__inference_batch_normalization_3_layer_call_fn_60445

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_587132
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ж
L
0__inference_module_wrapper_7_layer_call_fn_59640

args_0
identityЛ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_567522
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
Ј
е
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_59515

args_0A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@
identityѕбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOpЙ
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpг
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2
conv2d_1/BiasAddИ
IdentityIdentityconv2d_1/BiasAdd:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         !! : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         !! 
 
_user_specified_nameargs_0
В
M
1__inference_module_wrapper_11_layer_call_fn_59780

args_0
identityм
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_568102
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
А
h
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_59958

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/Constђ
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
ф
g
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_56775

args_0
identityp
activation_2/ReluReluargs_0*
T0*/
_output_shapes
:         @2
activation_2/Relu{
IdentityIdentityactivation_2/Relu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
═
Б
.__inference_module_wrapper_layer_call_fn_59403

args_0!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_576672
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameargs_0
Ќ
Ф
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_56822

args_0B
'conv2d_3_conv2d_readvariableop_resource:@ђ7
(conv2d_3_biasadd_readvariableop_resource:	ђ
identityѕбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOp▒
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_3/Conv2D/ReadVariableOp┐
conv2d_3/Conv2DConv2Dargs_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_3/Conv2Dе
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpГ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_3/BiasAdd╣
IdentityIdentityconv2d_3/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Чr
Н
E__inference_sequential_layer_call_and_return_conditional_losses_56969

inputs.
module_wrapper_56649: "
module_wrapper_56651: $
module_wrapper_2_56680: $
module_wrapper_2_56682: $
module_wrapper_2_56684: $
module_wrapper_2_56686: 0
module_wrapper_4_56707: @$
module_wrapper_4_56709:@$
module_wrapper_6_56738:@$
module_wrapper_6_56740:@$
module_wrapper_6_56742:@$
module_wrapper_6_56744:@0
module_wrapper_8_56765:@@$
module_wrapper_8_56767:@%
module_wrapper_10_56796:@%
module_wrapper_10_56798:@%
module_wrapper_10_56800:@%
module_wrapper_10_56802:@2
module_wrapper_12_56823:@ђ&
module_wrapper_12_56825:	ђ&
module_wrapper_14_56854:	ђ&
module_wrapper_14_56856:	ђ&
module_wrapper_14_56858:	ђ&
module_wrapper_14_56860:	ђ+
module_wrapper_18_56896:
ђђ&
module_wrapper_18_56898:	ђ&
module_wrapper_20_56929:	ђ&
module_wrapper_20_56931:	ђ&
module_wrapper_20_56933:	ђ&
module_wrapper_20_56935:	ђ*
module_wrapper_22_56956:	ђ%
module_wrapper_22_56958:
identityѕб&module_wrapper/StatefulPartitionedCallб)module_wrapper_10/StatefulPartitionedCallб)module_wrapper_12/StatefulPartitionedCallб)module_wrapper_14/StatefulPartitionedCallб)module_wrapper_18/StatefulPartitionedCallб(module_wrapper_2/StatefulPartitionedCallб)module_wrapper_20/StatefulPartitionedCallб)module_wrapper_22/StatefulPartitionedCallб(module_wrapper_4/StatefulPartitionedCallб(module_wrapper_6/StatefulPartitionedCallб(module_wrapper_8/StatefulPartitionedCallи
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_56649module_wrapper_56651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_566482(
&module_wrapper/StatefulPartitionedCallю
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_566592"
 module_wrapper_1/PartitionedCallў
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0module_wrapper_2_56680module_wrapper_2_56682module_wrapper_2_56684module_wrapper_2_56686*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_566792*
(module_wrapper_2/StatefulPartitionedCallъ
 module_wrapper_3/PartitionedCallPartitionedCall1module_wrapper_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_566942"
 module_wrapper_3/PartitionedCallС
(module_wrapper_4/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0module_wrapper_4_56707module_wrapper_4_56709*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_567062*
(module_wrapper_4/StatefulPartitionedCallъ
 module_wrapper_5/PartitionedCallPartitionedCall1module_wrapper_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_567172"
 module_wrapper_5/PartitionedCallў
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_5/PartitionedCall:output:0module_wrapper_6_56738module_wrapper_6_56740module_wrapper_6_56742module_wrapper_6_56744*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_567372*
(module_wrapper_6/StatefulPartitionedCallъ
 module_wrapper_7/PartitionedCallPartitionedCall1module_wrapper_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_567522"
 module_wrapper_7/PartitionedCallС
(module_wrapper_8/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_7/PartitionedCall:output:0module_wrapper_8_56765module_wrapper_8_56767*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_567642*
(module_wrapper_8/StatefulPartitionedCallъ
 module_wrapper_9/PartitionedCallPartitionedCall1module_wrapper_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_567752"
 module_wrapper_9/PartitionedCallЪ
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_56796module_wrapper_10_56798module_wrapper_10_56800module_wrapper_10_56802*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_567952+
)module_wrapper_10/StatefulPartitionedCallб
!module_wrapper_11/PartitionedCallPartitionedCall2module_wrapper_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_568102#
!module_wrapper_11/PartitionedCallв
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_11/PartitionedCall:output:0module_wrapper_12_56823module_wrapper_12_56825*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_568222+
)module_wrapper_12/StatefulPartitionedCallБ
!module_wrapper_13/PartitionedCallPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_568332#
!module_wrapper_13/PartitionedCallА
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_13/PartitionedCall:output:0module_wrapper_14_56854module_wrapper_14_56856module_wrapper_14_56858module_wrapper_14_56860*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_568532+
)module_wrapper_14/StatefulPartitionedCallБ
!module_wrapper_15/PartitionedCallPartitionedCall2module_wrapper_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_568682#
!module_wrapper_15/PartitionedCallЏ
!module_wrapper_16/PartitionedCallPartitionedCall*module_wrapper_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_568752#
!module_wrapper_16/PartitionedCallЊ
!module_wrapper_17/PartitionedCallPartitionedCall*module_wrapper_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_568832#
!module_wrapper_17/PartitionedCallс
)module_wrapper_18/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_17/PartitionedCall:output:0module_wrapper_18_56896module_wrapper_18_56898*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_568952+
)module_wrapper_18/StatefulPartitionedCallЏ
!module_wrapper_19/PartitionedCallPartitionedCall2module_wrapper_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_569062#
!module_wrapper_19/PartitionedCallЎ
)module_wrapper_20/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_19/PartitionedCall:output:0module_wrapper_20_56929module_wrapper_20_56931module_wrapper_20_56933module_wrapper_20_56935*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_569282+
)module_wrapper_20/StatefulPartitionedCallЏ
!module_wrapper_21/PartitionedCallPartitionedCall2module_wrapper_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_569432#
!module_wrapper_21/PartitionedCallР
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_21/PartitionedCall:output:0module_wrapper_22_56956module_wrapper_22_56958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_569552+
)module_wrapper_22/StatefulPartitionedCallџ
!module_wrapper_23/PartitionedCallPartitionedCall2module_wrapper_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_569662#
!module_wrapper_23/PartitionedCall█
IdentityIdentity*module_wrapper_23/PartitionedCall:output:0'^module_wrapper/StatefulPartitionedCall*^module_wrapper_10/StatefulPartitionedCall*^module_wrapper_12/StatefulPartitionedCall*^module_wrapper_14/StatefulPartitionedCall*^module_wrapper_18/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall*^module_wrapper_20/StatefulPartitionedCall*^module_wrapper_22/StatefulPartitionedCall)^module_wrapper_4/StatefulPartitionedCall)^module_wrapper_6/StatefulPartitionedCall)^module_wrapper_8/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2V
)module_wrapper_10/StatefulPartitionedCall)module_wrapper_10/StatefulPartitionedCall2V
)module_wrapper_12/StatefulPartitionedCall)module_wrapper_12/StatefulPartitionedCall2V
)module_wrapper_14/StatefulPartitionedCall)module_wrapper_14/StatefulPartitionedCall2V
)module_wrapper_18/StatefulPartitionedCall)module_wrapper_18/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall2V
)module_wrapper_20/StatefulPartitionedCall)module_wrapper_20/StatefulPartitionedCall2V
)module_wrapper_22/StatefulPartitionedCall)module_wrapper_22/StatefulPartitionedCall2T
(module_wrapper_4/StatefulPartitionedCall(module_wrapper_4/StatefulPartitionedCall2T
(module_wrapper_6/StatefulPartitionedCall(module_wrapper_6/StatefulPartitionedCall2T
(module_wrapper_8/StatefulPartitionedCall(module_wrapper_8/StatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
у
ѓ
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_59459

args_09
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб"batch_normalization/AssignNewValueб$batch_normalization/AssignNewValue_1б3batch_normalization/FusedBatchNormV3/ReadVariableOpб5batch_normalization/FusedBatchNormV3/ReadVariableOp_1б"batch_normalization/ReadVariableOpб$batch_normalization/ReadVariableOp_1░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpХ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1с
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpж
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1л
$batch_normalization/FusedBatchNormV3FusedBatchNormV3args_0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<2&
$batch_normalization/FusedBatchNormV3д
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue▓
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1і
IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         dd : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_1:W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
ж
Ж
*__inference_sequential_layer_call_fn_57973
module_wrapper_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@ђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ

unknown_21:	ђ

unknown_22:	ђ

unknown_23:
ђђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ

unknown_27:	ђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:
identityѕбStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_578372
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:         dd
.
_user_specified_namemodule_wrapper_input
Ј
h
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_57178

args_0
identityi
activation_4/ReluReluargs_0*
T0*(
_output_shapes
:         ђ2
activation_4/Relut
IdentityIdentityactivation_4/Relu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ь
g
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_59635

args_0
identity▓
max_pooling2d_1/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool|
IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
з
h
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_59915

args_0
identity│
max_pooling2d_3/MaxPoolMaxPoolargs_0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool}
IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
ѕ
Ў
N__inference_batch_normalization_layer_call_and_return_conditional_losses_60215

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Л
I
-__inference_max_pooling2d_layer_call_fn_58371

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_583652
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Йv
Г
E__inference_sequential_layer_call_and_return_conditional_losses_57837

inputs.
module_wrapper_57748: "
module_wrapper_57750: $
module_wrapper_2_57754: $
module_wrapper_2_57756: $
module_wrapper_2_57758: $
module_wrapper_2_57760: 0
module_wrapper_4_57764: @$
module_wrapper_4_57766:@$
module_wrapper_6_57770:@$
module_wrapper_6_57772:@$
module_wrapper_6_57774:@$
module_wrapper_6_57776:@0
module_wrapper_8_57780:@@$
module_wrapper_8_57782:@%
module_wrapper_10_57786:@%
module_wrapper_10_57788:@%
module_wrapper_10_57790:@%
module_wrapper_10_57792:@2
module_wrapper_12_57796:@ђ&
module_wrapper_12_57798:	ђ&
module_wrapper_14_57802:	ђ&
module_wrapper_14_57804:	ђ&
module_wrapper_14_57806:	ђ&
module_wrapper_14_57808:	ђ+
module_wrapper_18_57814:
ђђ&
module_wrapper_18_57816:	ђ&
module_wrapper_20_57820:	ђ&
module_wrapper_20_57822:	ђ&
module_wrapper_20_57824:	ђ&
module_wrapper_20_57826:	ђ*
module_wrapper_22_57830:	ђ%
module_wrapper_22_57832:
identityѕб&module_wrapper/StatefulPartitionedCallб)module_wrapper_10/StatefulPartitionedCallб)module_wrapper_12/StatefulPartitionedCallб)module_wrapper_14/StatefulPartitionedCallб)module_wrapper_16/StatefulPartitionedCallб)module_wrapper_18/StatefulPartitionedCallб(module_wrapper_2/StatefulPartitionedCallб)module_wrapper_20/StatefulPartitionedCallб)module_wrapper_21/StatefulPartitionedCallб)module_wrapper_22/StatefulPartitionedCallб(module_wrapper_4/StatefulPartitionedCallб(module_wrapper_6/StatefulPartitionedCallб(module_wrapper_8/StatefulPartitionedCallи
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_57748module_wrapper_57750*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_576672(
&module_wrapper/StatefulPartitionedCallю
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_576422"
 module_wrapper_1/PartitionedCallќ
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0module_wrapper_2_57754module_wrapper_2_57756module_wrapper_2_57758module_wrapper_2_57760*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_576182*
(module_wrapper_2/StatefulPartitionedCallъ
 module_wrapper_3/PartitionedCallPartitionedCall1module_wrapper_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_575812"
 module_wrapper_3/PartitionedCallС
(module_wrapper_4/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0module_wrapper_4_57764module_wrapper_4_57766*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_575612*
(module_wrapper_4/StatefulPartitionedCallъ
 module_wrapper_5/PartitionedCallPartitionedCall1module_wrapper_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_575362"
 module_wrapper_5/PartitionedCallќ
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_5/PartitionedCall:output:0module_wrapper_6_57770module_wrapper_6_57772module_wrapper_6_57774module_wrapper_6_57776*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !!@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_575122*
(module_wrapper_6/StatefulPartitionedCallъ
 module_wrapper_7/PartitionedCallPartitionedCall1module_wrapper_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_574752"
 module_wrapper_7/PartitionedCallС
(module_wrapper_8/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_7/PartitionedCall:output:0module_wrapper_8_57780module_wrapper_8_57782*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_574552*
(module_wrapper_8/StatefulPartitionedCallъ
 module_wrapper_9/PartitionedCallPartitionedCall1module_wrapper_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_574302"
 module_wrapper_9/PartitionedCallЮ
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_57786module_wrapper_10_57788module_wrapper_10_57790module_wrapper_10_57792*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_574062+
)module_wrapper_10/StatefulPartitionedCallб
!module_wrapper_11/PartitionedCallPartitionedCall2module_wrapper_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_573692#
!module_wrapper_11/PartitionedCallв
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_11/PartitionedCall:output:0module_wrapper_12_57796module_wrapper_12_57798*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_573492+
)module_wrapper_12/StatefulPartitionedCallБ
!module_wrapper_13/PartitionedCallPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_573242#
!module_wrapper_13/PartitionedCallЪ
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_13/PartitionedCall:output:0module_wrapper_14_57802module_wrapper_14_57804module_wrapper_14_57806module_wrapper_14_57808*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_573002+
)module_wrapper_14/StatefulPartitionedCallБ
!module_wrapper_15/PartitionedCallPartitionedCall2module_wrapper_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_572632#
!module_wrapper_15/PartitionedCall│
)module_wrapper_16/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_572472+
)module_wrapper_16/StatefulPartitionedCallЏ
!module_wrapper_17/PartitionedCallPartitionedCall2module_wrapper_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_572242#
!module_wrapper_17/PartitionedCallс
)module_wrapper_18/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_17/PartitionedCall:output:0module_wrapper_18_57814module_wrapper_18_57816*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_572032+
)module_wrapper_18/StatefulPartitionedCallЏ
!module_wrapper_19/PartitionedCallPartitionedCall2module_wrapper_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_571782#
!module_wrapper_19/PartitionedCallЌ
)module_wrapper_20/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_19/PartitionedCall:output:0module_wrapper_20_57820module_wrapper_20_57822module_wrapper_20_57824module_wrapper_20_57826*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_571542+
)module_wrapper_20/StatefulPartitionedCall▀
)module_wrapper_21/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_20/StatefulPartitionedCall:output:0*^module_wrapper_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_571012+
)module_wrapper_21/StatefulPartitionedCallЖ
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_21/StatefulPartitionedCall:output:0module_wrapper_22_57830module_wrapper_22_57832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_570742+
)module_wrapper_22/StatefulPartitionedCallџ
!module_wrapper_23/PartitionedCallPartitionedCall2module_wrapper_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_570492#
!module_wrapper_23/PartitionedCall│
IdentityIdentity*module_wrapper_23/PartitionedCall:output:0'^module_wrapper/StatefulPartitionedCall*^module_wrapper_10/StatefulPartitionedCall*^module_wrapper_12/StatefulPartitionedCall*^module_wrapper_14/StatefulPartitionedCall*^module_wrapper_16/StatefulPartitionedCall*^module_wrapper_18/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall*^module_wrapper_20/StatefulPartitionedCall*^module_wrapper_21/StatefulPartitionedCall*^module_wrapper_22/StatefulPartitionedCall)^module_wrapper_4/StatefulPartitionedCall)^module_wrapper_6/StatefulPartitionedCall)^module_wrapper_8/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2V
)module_wrapper_10/StatefulPartitionedCall)module_wrapper_10/StatefulPartitionedCall2V
)module_wrapper_12/StatefulPartitionedCall)module_wrapper_12/StatefulPartitionedCall2V
)module_wrapper_14/StatefulPartitionedCall)module_wrapper_14/StatefulPartitionedCall2V
)module_wrapper_16/StatefulPartitionedCall)module_wrapper_16/StatefulPartitionedCall2V
)module_wrapper_18/StatefulPartitionedCall)module_wrapper_18/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall2V
)module_wrapper_20/StatefulPartitionedCall)module_wrapper_20/StatefulPartitionedCall2V
)module_wrapper_21/StatefulPartitionedCall)module_wrapper_21/StatefulPartitionedCall2V
)module_wrapper_22/StatefulPartitionedCall)module_wrapper_22/StatefulPartitionedCall2T
(module_wrapper_4/StatefulPartitionedCall(module_wrapper_4/StatefulPartitionedCall2T
(module_wrapper_6/StatefulPartitionedCall(module_wrapper_6/StatefulPartitionedCall2T
(module_wrapper_8/StatefulPartitionedCall(module_wrapper_8/StatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Й
┐
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_60295

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
№
h
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_57369

args_0
identity▓
max_pooling2d_2/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool|
IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Њ
h
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_57049

args_0
identityq
activation_5/SoftmaxSoftmaxargs_0*
T0*'
_output_shapes
:         2
activation_5/Softmaxr
IdentityIdentityactivation_5/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameargs_0
Ю
л
1__inference_module_wrapper_14_layer_call_fn_59905

args_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_573002
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
Ј
е
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_56706

args_0A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@
identityѕбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOpЙ
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpг
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2
conv2d_1/BiasAddИ
IdentityIdentityconv2d_1/BiasAdd:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         !!@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         !! : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         !! 
 
_user_specified_nameargs_0
Й
k
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_57247

args_0
identityѕs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/dropout/Constћ
dropout/dropout/MulMulargs_0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout/dropout/Muld
dropout/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout/dropout/ShapeН
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2 
dropout/dropout/GreaterEqual/yу
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2
dropout/dropout/GreaterEqualа
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout/dropout/CastБ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout/dropout/Mul_1v
IdentityIdentitydropout/dropout/Mul_1:z:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
ѕ
Ў
N__inference_batch_normalization_layer_call_and_return_conditional_losses_58255

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╝
й
N__inference_batch_normalization_layer_call_and_return_conditional_losses_60233

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
▓
А
1__inference_module_wrapper_18_layer_call_fn_60012

args_0
unknown:
ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_572032
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
т
л
5__inference_batch_normalization_1_layer_call_fn_60321

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_584372
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ц
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_59413

args_0
identityl
activation/ReluReluargs_0*
T0*/
_output_shapes
:         dd 2
activation/Reluy
IdentityIdentityactivation/Relu:activations:0*
T0*/
_output_shapes
:         dd 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
»
h
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_59828

args_0
identityq
activation_3/ReluReluargs_0*
T0*0
_output_shapes
:         ђ2
activation_3/Relu|
IdentityIdentityactivation_3/Relu:activations:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameargs_0
і
Џ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_58531

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
У
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_56694

args_0
identity«
max_pooling2d/MaxPoolMaxPoolargs_0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolz
IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
Й
┐
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_60357

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
┌
Ќ
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_57406

args_0;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб$batch_normalization_2/AssignNewValueб&batch_normalization_2/AssignNewValue_1б5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1Х
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1ж
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1▄
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_2/FusedBatchNormV3░
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue╝
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1ў
IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : : : 2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_1:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ю
ъ
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_60149

args_09
&dense_1_matmul_readvariableop_resource:	ђ5
'dense_1_biasadd_readvariableop_resource:
identityѕбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
dense_1/MatMul/ReadVariableOpІ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddГ
IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameargs_0
╠
M
1__inference_module_wrapper_23_layer_call_fn_60192

args_0
identity╩
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_569662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameargs_0
ф
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_58641

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
џ
Ъ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_60401

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Њ
h
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_60182

args_0
identityq
activation_5/SoftmaxSoftmaxargs_0*
T0*'
_output_shapes
:         2
activation_5/Softmaxr
IdentityIdentityactivation_5/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameargs_0
Ж
L
0__inference_module_wrapper_9_layer_call_fn_59698

args_0
identityЛ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_567752
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
У
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_59490

args_0
identity«
max_pooling2d/MaxPoolMaxPoolargs_0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolz
IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
Ь
g
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_59630

args_0
identity▓
max_pooling2d_1/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool|
IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         !!@:W S
/
_output_shapes
:         !!@
 
_user_specified_nameargs_0
Ж
L
0__inference_module_wrapper_3_layer_call_fn_59505

args_0
identityЛ
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_575812
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd :W S
/
_output_shapes
:         dd 
 
_user_specified_nameargs_0
і
Џ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_60277

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╬
├
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_58713

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*о
serving_default┬
]
module_wrapper_inputE
&serving_default_module_wrapper_input:0         ddE
module_wrapper_230
StatefulPartitionedCall:0         tensorflow/serving/predict:ом
╠ 
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer_with_weights-9
layer-20
layer-21
layer_with_weights-10
layer-22
layer-23
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+Х&call_and_return_all_conditional_losses
и__call__
И_default_save_signature"Ў
_tf_keras_sequentialЩ{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "module_wrapper_input"}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}]}, "shared_object_id": 1, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "float32", "module_wrapper_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 2}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
╗
_module
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+╣&call_and_return_all_conditional_losses
║__call__"Ю
_tf_keras_layerЃ{"name": "module_wrapper", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
й
$_module
%	variables
&regularization_losses
'trainable_variables
(	keras_api
+╗&call_and_return_all_conditional_losses
╝__call__"Ъ
_tf_keras_layerЁ{"name": "module_wrapper_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
й
)_module
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+й&call_and_return_all_conditional_losses
Й__call__"Ъ
_tf_keras_layerЁ{"name": "module_wrapper_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
й
._module
/	variables
0regularization_losses
1trainable_variables
2	keras_api
+┐&call_and_return_all_conditional_losses
└__call__"Ъ
_tf_keras_layerЁ{"name": "module_wrapper_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
й
3_module
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+┴&call_and_return_all_conditional_losses
┬__call__"Ъ
_tf_keras_layerЁ{"name": "module_wrapper_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
й
8_module
9	variables
:regularization_losses
;trainable_variables
<	keras_api
+├&call_and_return_all_conditional_losses
─__call__"Ъ
_tf_keras_layerЁ{"name": "module_wrapper_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
й
=_module
>	variables
?regularization_losses
@trainable_variables
A	keras_api
+┼&call_and_return_all_conditional_losses
к__call__"Ъ
_tf_keras_layerЁ{"name": "module_wrapper_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
й
B_module
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
+К&call_and_return_all_conditional_losses
╚__call__"Ъ
_tf_keras_layerЁ{"name": "module_wrapper_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
й
G_module
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
+╔&call_and_return_all_conditional_losses
╩__call__"Ъ
_tf_keras_layerЁ{"name": "module_wrapper_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
й
L_module
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
+╦&call_and_return_all_conditional_losses
╠__call__"Ъ
_tf_keras_layerЁ{"name": "module_wrapper_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
Й
Q_module
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+═&call_and_return_all_conditional_losses
╬__call__"а
_tf_keras_layerє{"name": "module_wrapper_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
Й
V_module
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
+¤&call_and_return_all_conditional_losses
л__call__"а
_tf_keras_layerє{"name": "module_wrapper_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
Й
[_module
\	variables
]regularization_losses
^trainable_variables
_	keras_api
+Л&call_and_return_all_conditional_losses
м__call__"а
_tf_keras_layerє{"name": "module_wrapper_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
Й
`_module
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
+М&call_and_return_all_conditional_losses
н__call__"а
_tf_keras_layerє{"name": "module_wrapper_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
Й
e_module
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+Н&call_and_return_all_conditional_losses
о__call__"а
_tf_keras_layerє{"name": "module_wrapper_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
Й
j_module
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
+О&call_and_return_all_conditional_losses
п__call__"а
_tf_keras_layerє{"name": "module_wrapper_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
Й
o_module
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
+┘&call_and_return_all_conditional_losses
┌__call__"а
_tf_keras_layerє{"name": "module_wrapper_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
Й
t_module
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
+█&call_and_return_all_conditional_losses
▄__call__"а
_tf_keras_layerє{"name": "module_wrapper_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
Й
y_module
z	variables
{regularization_losses
|trainable_variables
}	keras_api
+П&call_and_return_all_conditional_losses
я__call__"а
_tf_keras_layerє{"name": "module_wrapper_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
┴
~_module
	variables
ђregularization_losses
Ђtrainable_variables
ѓ	keras_api
+▀&call_and_return_all_conditional_losses
Я__call__"а
_tf_keras_layerє{"name": "module_wrapper_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
├
Ѓ_module
ё	variables
Ёregularization_losses
єtrainable_variables
Є	keras_api
+р&call_and_return_all_conditional_losses
Р__call__"а
_tf_keras_layerє{"name": "module_wrapper_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
├
ѕ_module
Ѕ	variables
іregularization_losses
Іtrainable_variables
ї	keras_api
+с&call_and_return_all_conditional_losses
С__call__"а
_tf_keras_layerє{"name": "module_wrapper_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
├
Ї_module
ј	variables
Јregularization_losses
љtrainable_variables
Љ	keras_api
+т&call_and_return_all_conditional_losses
Т__call__"а
_tf_keras_layerє{"name": "module_wrapper_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
├
њ_module
Њ	variables
ћregularization_losses
Ћtrainable_variables
ќ	keras_api
+у&call_and_return_all_conditional_losses
У__call__"а
_tf_keras_layerє{"name": "module_wrapper_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
┘
	Ќiter

ўdecay
Ўlearning_rate
џmomentumЏmomentumаюmomentumАЮmomentumбъmomentumБАmomentumцбmomentumЦБmomentumдцmomentumДДmomentumееmomentumЕЕmomentumффmomentumФГmomentumг«momentumГ»momentum«░momentum»│momentum░┤momentum▒хmomentum▓Хmomentum│╣momentum┤║momentumх"
	optimizer
Х
Џ0
ю1
Ю2
ъ3
Ъ4
а5
А6
б7
Б8
ц9
Ц10
д11
Д12
е13
Е14
ф15
Ф16
г17
Г18
«19
»20
░21
▒22
▓23
│24
┤25
х26
Х27
и28
И29
╣30
║31"
trackable_list_wrapper
 "
trackable_list_wrapper
▄
Џ0
ю1
Ю2
ъ3
А4
б5
Б6
ц7
Д8
е9
Е10
ф11
Г12
«13
»14
░15
│16
┤17
х18
Х19
╣20
║21"
trackable_list_wrapper
М
	variables
regularization_losses
╗non_trainable_variables
trainable_variables
╝layers
йmetrics
Йlayer_metrics
 ┐layer_regularization_losses
и__call__
И_default_save_signature
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
-
жserving_default"
signature_map
§

Џkernel
	юbias
└	variables
┴regularization_losses
┬trainable_variables
├	keras_api
+Ж&call_and_return_all_conditional_losses
в__call__"л	
_tf_keras_layerХ	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 3]}}
0
Џ0
ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Џ0
ю1"
trackable_list_wrapper
х
 	variables
!regularization_losses
─non_trainable_variables
"trainable_variables
┼layers
кmetrics
Кlayer_metrics
 ╚layer_regularization_losses
║__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
О
╔	variables
╩regularization_losses
╦trainable_variables
╠	keras_api
+В&call_and_return_all_conditional_losses
ь__call__"┬
_tf_keras_layerе{"name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
%	variables
&regularization_losses
═non_trainable_variables
'trainable_variables
╬layers
¤metrics
лlayer_metrics
 Лlayer_regularization_losses
╝__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
├	
	мaxis

Юgamma
	ъbeta
Ъmoving_mean
аmoving_variance
М	variables
нregularization_losses
Нtrainable_variables
о	keras_api
+Ь&call_and_return_all_conditional_losses
№__call__"С
_tf_keras_layer╩{"name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 32]}}
@
Ю0
ъ1
Ъ2
а3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ю0
ъ1"
trackable_list_wrapper
х
*	variables
+regularization_losses
Оnon_trainable_variables
,trainable_variables
пlayers
┘metrics
┌layer_metrics
 █layer_regularization_losses
Й__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
Ђ
▄	variables
Пregularization_losses
яtrainable_variables
▀	keras_api
+­&call_and_return_all_conditional_losses
ы__call__"В
_tf_keras_layerм{"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
/	variables
0regularization_losses
Яnon_trainable_variables
1trainable_variables
рlayers
Рmetrics
сlayer_metrics
 Сlayer_regularization_losses
└__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
Ч	
Аkernel
	бbias
т	variables
Тregularization_losses
уtrainable_variables
У	keras_api
+Ы&call_and_return_all_conditional_losses
з__call__"¤
_tf_keras_layerх{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 32]}}
0
А0
б1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
А0
б1"
trackable_list_wrapper
х
4	variables
5regularization_losses
жnon_trainable_variables
6trainable_variables
Жlayers
вmetrics
Вlayer_metrics
 ьlayer_regularization_losses
┬__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
█
Ь	variables
№regularization_losses
­trainable_variables
ы	keras_api
+З&call_and_return_all_conditional_losses
ш__call__"к
_tf_keras_layerг{"name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
9	variables
:regularization_losses
Ыnon_trainable_variables
;trainable_variables
зlayers
Зmetrics
шlayer_metrics
 Шlayer_regularization_losses
─__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
┼	
	эaxis

Бgamma
	цbeta
Цmoving_mean
дmoving_variance
Э	variables
щregularization_losses
Щtrainable_variables
ч	keras_api
+Ш&call_and_return_all_conditional_losses
э__call__"Т
_tf_keras_layer╠{"name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 64]}}
@
Б0
ц1
Ц2
д3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Б0
ц1"
trackable_list_wrapper
х
>	variables
?regularization_losses
Чnon_trainable_variables
@trainable_variables
§layers
■metrics
 layer_metrics
 ђlayer_regularization_losses
к__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
Ё
Ђ	variables
ѓregularization_losses
Ѓtrainable_variables
ё	keras_api
+Э&call_and_return_all_conditional_losses
щ__call__"­
_tf_keras_layerо{"name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
C	variables
Dregularization_losses
Ёnon_trainable_variables
Etrainable_variables
єlayers
Єmetrics
ѕlayer_metrics
 Ѕlayer_regularization_losses
╚__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
Ч	
Дkernel
	еbias
і	variables
Іregularization_losses
їtrainable_variables
Ї	keras_api
+Щ&call_and_return_all_conditional_losses
ч__call__"¤
_tf_keras_layerх{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
0
Д0
е1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Д0
е1"
trackable_list_wrapper
х
H	variables
Iregularization_losses
јnon_trainable_variables
Jtrainable_variables
Јlayers
љmetrics
Љlayer_metrics
 њlayer_regularization_losses
╩__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
█
Њ	variables
ћregularization_losses
Ћtrainable_variables
ќ	keras_api
+Ч&call_and_return_all_conditional_losses
§__call__"к
_tf_keras_layerг{"name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
M	variables
Nregularization_losses
Ќnon_trainable_variables
Otrainable_variables
ўlayers
Ўmetrics
џlayer_metrics
 Џlayer_regularization_losses
╠__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
┼	
	юaxis

Еgamma
	фbeta
Фmoving_mean
гmoving_variance
Ю	variables
ъregularization_losses
Ъtrainable_variables
а	keras_api
+■&call_and_return_all_conditional_losses
 __call__"Т
_tf_keras_layer╠{"name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
@
Е0
ф1
Ф2
г3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Е0
ф1"
trackable_list_wrapper
х
R	variables
Sregularization_losses
Аnon_trainable_variables
Ttrainable_variables
бlayers
Бmetrics
цlayer_metrics
 Цlayer_regularization_losses
╬__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
Ё
д	variables
Дregularization_losses
еtrainable_variables
Е	keras_api
+ђ&call_and_return_all_conditional_losses
Ђ__call__"­
_tf_keras_layerо{"name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
W	variables
Xregularization_losses
фnon_trainable_variables
Ytrainable_variables
Фlayers
гmetrics
Гlayer_metrics
 «layer_regularization_losses
л__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
ч	
Гkernel
	«bias
»	variables
░regularization_losses
▒trainable_variables
▓	keras_api
+ѓ&call_and_return_all_conditional_losses
Ѓ__call__"╬
_tf_keras_layer┤{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
0
Г0
«1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Г0
«1"
trackable_list_wrapper
х
\	variables
]regularization_losses
│non_trainable_variables
^trainable_variables
┤layers
хmetrics
Хlayer_metrics
 иlayer_regularization_losses
м__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
█
И	variables
╣regularization_losses
║trainable_variables
╗	keras_api
+ё&call_and_return_all_conditional_losses
Ё__call__"к
_tf_keras_layerг{"name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
a	variables
bregularization_losses
╝non_trainable_variables
ctrainable_variables
йlayers
Йmetrics
┐layer_metrics
 └layer_regularization_losses
н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
┼	
	┴axis

»gamma
	░beta
▒moving_mean
▓moving_variance
┬	variables
├regularization_losses
─trainable_variables
┼	keras_api
+є&call_and_return_all_conditional_losses
Є__call__"Т
_tf_keras_layer╠{"name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
@
»0
░1
▒2
▓3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
»0
░1"
trackable_list_wrapper
х
f	variables
gregularization_losses
кnon_trainable_variables
htrainable_variables
Кlayers
╚metrics
╔layer_metrics
 ╩layer_regularization_losses
о__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
Ё
╦	variables
╠regularization_losses
═trainable_variables
╬	keras_api
+ѕ&call_and_return_all_conditional_losses
Ѕ__call__"­
_tf_keras_layerо{"name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
k	variables
lregularization_losses
¤non_trainable_variables
mtrainable_variables
лlayers
Лmetrics
мlayer_metrics
 Мlayer_regularization_losses
п__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
у
н	variables
Нregularization_losses
оtrainable_variables
О	keras_api
+і&call_and_return_all_conditional_losses
І__call__"м
_tf_keras_layerИ{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
p	variables
qregularization_losses
пnon_trainable_variables
rtrainable_variables
┘layers
┌metrics
█layer_metrics
 ▄layer_regularization_losses
┌__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
У
П	variables
яregularization_losses
▀trainable_variables
Я	keras_api
+ї&call_and_return_all_conditional_losses
Ї__call__"М
_tf_keras_layer╣{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
u	variables
vregularization_losses
рnon_trainable_variables
wtrainable_variables
Рlayers
сmetrics
Сlayer_metrics
 тlayer_regularization_losses
▄__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
ч
│kernel
	┤bias
Т	variables
уregularization_losses
Уtrainable_variables
ж	keras_api
+ј&call_and_return_all_conditional_losses
Ј__call__"╬
_tf_keras_layer┤{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
0
│0
┤1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
│0
┤1"
trackable_list_wrapper
х
z	variables
{regularization_losses
Жnon_trainable_variables
|trainable_variables
вlayers
Вmetrics
ьlayer_metrics
 Ьlayer_regularization_losses
я__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
█
№	variables
­regularization_losses
ыtrainable_variables
Ы	keras_api
+љ&call_and_return_all_conditional_losses
Љ__call__"к
_tf_keras_layerг{"name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
и
	variables
ђregularization_losses
зnon_trainable_variables
Ђtrainable_variables
Зlayers
шmetrics
Шlayer_metrics
 эlayer_regularization_losses
Я__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
┐	
	Эaxis

хgamma
	Хbeta
иmoving_mean
Иmoving_variance
щ	variables
Щregularization_losses
чtrainable_variables
Ч	keras_api
+њ&call_and_return_all_conditional_losses
Њ__call__"Я
_tf_keras_layerк{"name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
@
х0
Х1
и2
И3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
х0
Х1"
trackable_list_wrapper
И
ё	variables
Ёregularization_losses
§non_trainable_variables
єtrainable_variables
■layers
 metrics
ђlayer_metrics
 Ђlayer_regularization_losses
Р__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
в
ѓ	variables
Ѓregularization_losses
ёtrainable_variables
Ё	keras_api
+ћ&call_and_return_all_conditional_losses
Ћ__call__"о
_tf_keras_layer╝{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѕ	variables
іregularization_losses
єnon_trainable_variables
Іtrainable_variables
Єlayers
ѕmetrics
Ѕlayer_metrics
 іlayer_regularization_losses
С__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
Ч
╣kernel
	║bias
І	variables
їregularization_losses
Їtrainable_variables
ј	keras_api
+ќ&call_and_return_all_conditional_losses
Ќ__call__"¤
_tf_keras_layerх{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 29, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
0
╣0
║1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
╣0
║1"
trackable_list_wrapper
И
ј	variables
Јregularization_losses
Јnon_trainable_variables
љtrainable_variables
љlayers
Љmetrics
њlayer_metrics
 Њlayer_regularization_losses
Т__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
я
ћ	variables
Ћregularization_losses
ќtrainable_variables
Ќ	keras_api
+ў&call_and_return_all_conditional_losses
Ў__call__"╔
_tf_keras_layer»{"name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "softmax"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Њ	variables
ћregularization_losses
ўnon_trainable_variables
Ћtrainable_variables
Ўlayers
џmetrics
Џlayer_metrics
 юlayer_regularization_losses
У__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
6:4 2module_wrapper/conv2d/kernel
(:& 2module_wrapper/conv2d/bias
8:6 2*module_wrapper_2/batch_normalization/gamma
7:5 2)module_wrapper_2/batch_normalization/beta
@:>  (20module_wrapper_2/batch_normalization/moving_mean
D:B  (24module_wrapper_2/batch_normalization/moving_variance
::8 @2 module_wrapper_4/conv2d_1/kernel
,:*@2module_wrapper_4/conv2d_1/bias
::8@2,module_wrapper_6/batch_normalization_1/gamma
9:7@2+module_wrapper_6/batch_normalization_1/beta
B:@@ (22module_wrapper_6/batch_normalization_1/moving_mean
F:D@ (26module_wrapper_6/batch_normalization_1/moving_variance
::8@@2 module_wrapper_8/conv2d_2/kernel
,:*@2module_wrapper_8/conv2d_2/bias
;:9@2-module_wrapper_10/batch_normalization_2/gamma
::8@2,module_wrapper_10/batch_normalization_2/beta
C:A@ (23module_wrapper_10/batch_normalization_2/moving_mean
G:E@ (27module_wrapper_10/batch_normalization_2/moving_variance
<::@ђ2!module_wrapper_12/conv2d_3/kernel
.:,ђ2module_wrapper_12/conv2d_3/bias
<::ђ2-module_wrapper_14/batch_normalization_3/gamma
;:9ђ2,module_wrapper_14/batch_normalization_3/beta
D:Bђ (23module_wrapper_14/batch_normalization_3/moving_mean
H:Fђ (27module_wrapper_14/batch_normalization_3/moving_variance
2:0
ђђ2module_wrapper_18/dense/kernel
+:)ђ2module_wrapper_18/dense/bias
<::ђ2-module_wrapper_20/batch_normalization_4/gamma
;:9ђ2,module_wrapper_20/batch_normalization_4/beta
D:Bђ (23module_wrapper_20/batch_normalization_4/moving_mean
H:Fђ (27module_wrapper_20/batch_normalization_4/moving_variance
3:1	ђ2 module_wrapper_22/dense_1/kernel
,:*2module_wrapper_22/dense_1/bias
p
Ъ0
а1
Ц2
д3
Ф4
г5
▒6
▓7
и8
И9"
trackable_list_wrapper
о
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23"
trackable_list_wrapper
0
Ю0
ъ1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Џ0
ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Џ0
ю1"
trackable_list_wrapper
И
└	variables
┴regularization_losses
Ъnon_trainable_variables
┬trainable_variables
аlayers
Аmetrics
бlayer_metrics
 Бlayer_regularization_losses
в__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
╔	variables
╩regularization_losses
цnon_trainable_variables
╦trainable_variables
Цlayers
дmetrics
Дlayer_metrics
 еlayer_regularization_losses
ь__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
Ю0
ъ1
Ъ2
а3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ю0
ъ1"
trackable_list_wrapper
И
М	variables
нregularization_losses
Еnon_trainable_variables
Нtrainable_variables
фlayers
Фmetrics
гlayer_metrics
 Гlayer_regularization_losses
№__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
0
Ъ0
а1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
▄	variables
Пregularization_losses
«non_trainable_variables
яtrainable_variables
»layers
░metrics
▒layer_metrics
 ▓layer_regularization_losses
ы__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
А0
б1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
А0
б1"
trackable_list_wrapper
И
т	variables
Тregularization_losses
│non_trainable_variables
уtrainable_variables
┤layers
хmetrics
Хlayer_metrics
 иlayer_regularization_losses
з__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ь	variables
№regularization_losses
Иnon_trainable_variables
­trainable_variables
╣layers
║metrics
╗layer_metrics
 ╝layer_regularization_losses
ш__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
Б0
ц1
Ц2
д3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Б0
ц1"
trackable_list_wrapper
И
Э	variables
щregularization_losses
йnon_trainable_variables
Щtrainable_variables
Йlayers
┐metrics
└layer_metrics
 ┴layer_regularization_losses
э__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
0
Ц0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ђ	variables
ѓregularization_losses
┬non_trainable_variables
Ѓtrainable_variables
├layers
─metrics
┼layer_metrics
 кlayer_regularization_losses
щ__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Д0
е1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Д0
е1"
trackable_list_wrapper
И
і	variables
Іregularization_losses
Кnon_trainable_variables
їtrainable_variables
╚layers
╔metrics
╩layer_metrics
 ╦layer_regularization_losses
ч__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Њ	variables
ћregularization_losses
╠non_trainable_variables
Ћtrainable_variables
═layers
╬metrics
¤layer_metrics
 лlayer_regularization_losses
§__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
Е0
ф1
Ф2
г3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Е0
ф1"
trackable_list_wrapper
И
Ю	variables
ъregularization_losses
Лnon_trainable_variables
Ъtrainable_variables
мlayers
Мmetrics
нlayer_metrics
 Нlayer_regularization_losses
 __call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
0
Ф0
г1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
д	variables
Дregularization_losses
оnon_trainable_variables
еtrainable_variables
Оlayers
пmetrics
┘layer_metrics
 ┌layer_regularization_losses
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Г0
«1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Г0
«1"
trackable_list_wrapper
И
»	variables
░regularization_losses
█non_trainable_variables
▒trainable_variables
▄layers
Пmetrics
яlayer_metrics
 ▀layer_regularization_losses
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
И	variables
╣regularization_losses
Яnon_trainable_variables
║trainable_variables
рlayers
Рmetrics
сlayer_metrics
 Сlayer_regularization_losses
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
»0
░1
▒2
▓3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
»0
░1"
trackable_list_wrapper
И
┬	variables
├regularization_losses
тnon_trainable_variables
─trainable_variables
Тlayers
уmetrics
Уlayer_metrics
 жlayer_regularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
0
▒0
▓1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
╦	variables
╠regularization_losses
Жnon_trainable_variables
═trainable_variables
вlayers
Вmetrics
ьlayer_metrics
 Ьlayer_regularization_losses
Ѕ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
н	variables
Нregularization_losses
№non_trainable_variables
оtrainable_variables
­layers
ыmetrics
Ыlayer_metrics
 зlayer_regularization_losses
І__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
П	variables
яregularization_losses
Зnon_trainable_variables
▀trainable_variables
шlayers
Шmetrics
эlayer_metrics
 Эlayer_regularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
│0
┤1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
│0
┤1"
trackable_list_wrapper
И
Т	variables
уregularization_losses
щnon_trainable_variables
Уtrainable_variables
Щlayers
чmetrics
Чlayer_metrics
 §layer_regularization_losses
Ј__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
№	variables
­regularization_losses
■non_trainable_variables
ыtrainable_variables
 layers
ђmetrics
Ђlayer_metrics
 ѓlayer_regularization_losses
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
х0
Х1
и2
И3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
х0
Х1"
trackable_list_wrapper
И
щ	variables
Щregularization_losses
Ѓnon_trainable_variables
чtrainable_variables
ёlayers
Ёmetrics
єlayer_metrics
 Єlayer_regularization_losses
Њ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
0
и0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѓ	variables
Ѓregularization_losses
ѕnon_trainable_variables
ёtrainable_variables
Ѕlayers
іmetrics
Іlayer_metrics
 їlayer_regularization_losses
Ћ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
╣0
║1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
╣0
║1"
trackable_list_wrapper
И
І	variables
їregularization_losses
Їnon_trainable_variables
Їtrainable_variables
јlayers
Јmetrics
љlayer_metrics
 Љlayer_regularization_losses
Ќ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ћ	variables
Ћregularization_losses
њnon_trainable_variables
ќtrainable_variables
Њlayers
ћmetrics
Ћlayer_metrics
 ќlayer_regularization_losses
Ў__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
О

Ќtotal

ўcount
Ў	variables
џ	keras_api"ю
_tf_keras_metricЂ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 3}
Џ

Џtotal

юcount
Ю
_fn_kwargs
ъ	variables
Ъ	keras_api"¤
_tf_keras_metric┤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 2}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Ъ0
а1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Ц0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Ф0
г1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
▒0
▓1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
и0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
Ќ0
ў1"
trackable_list_wrapper
.
Ў	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Џ0
ю1"
trackable_list_wrapper
.
ъ	variables"
_generic_user_object
A:? 2)SGD/module_wrapper/conv2d/kernel/momentum
3:1 2'SGD/module_wrapper/conv2d/bias/momentum
C:A 27SGD/module_wrapper_2/batch_normalization/gamma/momentum
B:@ 26SGD/module_wrapper_2/batch_normalization/beta/momentum
E:C @2-SGD/module_wrapper_4/conv2d_1/kernel/momentum
7:5@2+SGD/module_wrapper_4/conv2d_1/bias/momentum
E:C@29SGD/module_wrapper_6/batch_normalization_1/gamma/momentum
D:B@28SGD/module_wrapper_6/batch_normalization_1/beta/momentum
E:C@@2-SGD/module_wrapper_8/conv2d_2/kernel/momentum
7:5@2+SGD/module_wrapper_8/conv2d_2/bias/momentum
F:D@2:SGD/module_wrapper_10/batch_normalization_2/gamma/momentum
E:C@29SGD/module_wrapper_10/batch_normalization_2/beta/momentum
G:E@ђ2.SGD/module_wrapper_12/conv2d_3/kernel/momentum
9:7ђ2,SGD/module_wrapper_12/conv2d_3/bias/momentum
G:Eђ2:SGD/module_wrapper_14/batch_normalization_3/gamma/momentum
F:Dђ29SGD/module_wrapper_14/batch_normalization_3/beta/momentum
=:;
ђђ2+SGD/module_wrapper_18/dense/kernel/momentum
6:4ђ2)SGD/module_wrapper_18/dense/bias/momentum
G:Eђ2:SGD/module_wrapper_20/batch_normalization_4/gamma/momentum
F:Dђ29SGD/module_wrapper_20/batch_normalization_4/beta/momentum
>:<	ђ2-SGD/module_wrapper_22/dense_1/kernel/momentum
7:52+SGD/module_wrapper_22/dense_1/bias/momentum
Р2▀
E__inference_sequential_layer_call_and_return_conditional_losses_59073
E__inference_sequential_layer_call_and_return_conditional_losses_59227
E__inference_sequential_layer_call_and_return_conditional_losses_58065
E__inference_sequential_layer_call_and_return_conditional_losses_58157└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ш2з
*__inference_sequential_layer_call_fn_57036
*__inference_sequential_layer_call_fn_59296
*__inference_sequential_layer_call_fn_59365
*__inference_sequential_layer_call_fn_57973└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
 __inference__wrapped_model_56631╦
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *;б8
6і3
module_wrapper_input         dd
▄2┘
I__inference_module_wrapper_layer_call_and_return_conditional_losses_59375
I__inference_module_wrapper_layer_call_and_return_conditional_losses_59385└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
д2Б
.__inference_module_wrapper_layer_call_fn_59394
.__inference_module_wrapper_layer_call_fn_59403└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Я2П
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_59408
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_59413└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ф2Д
0__inference_module_wrapper_1_layer_call_fn_59418
0__inference_module_wrapper_1_layer_call_fn_59423└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Я2П
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_59441
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_59459└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ф2Д
0__inference_module_wrapper_2_layer_call_fn_59472
0__inference_module_wrapper_2_layer_call_fn_59485└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Я2П
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_59490
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_59495└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ф2Д
0__inference_module_wrapper_3_layer_call_fn_59500
0__inference_module_wrapper_3_layer_call_fn_59505└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Я2П
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_59515
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_59525└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ф2Д
0__inference_module_wrapper_4_layer_call_fn_59534
0__inference_module_wrapper_4_layer_call_fn_59543└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Я2П
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_59548
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_59553└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ф2Д
0__inference_module_wrapper_5_layer_call_fn_59558
0__inference_module_wrapper_5_layer_call_fn_59563└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Я2П
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_59581
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_59599└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ф2Д
0__inference_module_wrapper_6_layer_call_fn_59612
0__inference_module_wrapper_6_layer_call_fn_59625└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Я2П
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_59630
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_59635└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ф2Д
0__inference_module_wrapper_7_layer_call_fn_59640
0__inference_module_wrapper_7_layer_call_fn_59645└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Я2П
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_59655
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_59665└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ф2Д
0__inference_module_wrapper_8_layer_call_fn_59674
0__inference_module_wrapper_8_layer_call_fn_59683└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Я2П
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_59688
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_59693└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ф2Д
0__inference_module_wrapper_9_layer_call_fn_59698
0__inference_module_wrapper_9_layer_call_fn_59703└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_59721
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_59739└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_10_layer_call_fn_59752
1__inference_module_wrapper_10_layer_call_fn_59765└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_59770
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_59775└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_11_layer_call_fn_59780
1__inference_module_wrapper_11_layer_call_fn_59785└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_59795
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_59805└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_12_layer_call_fn_59814
1__inference_module_wrapper_12_layer_call_fn_59823└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_59828
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_59833└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_13_layer_call_fn_59838
1__inference_module_wrapper_13_layer_call_fn_59843└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_59861
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_59879└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_14_layer_call_fn_59892
1__inference_module_wrapper_14_layer_call_fn_59905└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_59910
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_59915└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_15_layer_call_fn_59920
1__inference_module_wrapper_15_layer_call_fn_59925└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_59930
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_59942└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_16_layer_call_fn_59947
1__inference_module_wrapper_16_layer_call_fn_59952└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_59958
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_59964└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_17_layer_call_fn_59969
1__inference_module_wrapper_17_layer_call_fn_59974└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_59984
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_59994└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_18_layer_call_fn_60003
1__inference_module_wrapper_18_layer_call_fn_60012└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_60017
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_60022└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_19_layer_call_fn_60027
1__inference_module_wrapper_19_layer_call_fn_60032└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_60052
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_60086└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_20_layer_call_fn_60099
1__inference_module_wrapper_20_layer_call_fn_60112└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_60117
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_60129└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_21_layer_call_fn_60134
1__inference_module_wrapper_21_layer_call_fn_60139└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_60149
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_60159└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_22_layer_call_fn_60168
1__inference_module_wrapper_22_layer_call_fn_60177└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Р2▀
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_60182
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_60187└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
г2Е
1__inference_module_wrapper_23_layer_call_fn_60192
1__inference_module_wrapper_23_layer_call_fn_60197└
и▓│
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ОBн
#__inference_signature_wrapper_58232module_wrapper_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
N__inference_batch_normalization_layer_call_and_return_conditional_losses_60215
N__inference_batch_normalization_layer_call_and_return_conditional_losses_60233┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ц2А
3__inference_batch_normalization_layer_call_fn_60246
3__inference_batch_normalization_layer_call_fn_60259┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
░2Г
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_58365Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ћ2њ
-__inference_max_pooling2d_layer_call_fn_58371Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
я2█
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_60277
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_60295┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
5__inference_batch_normalization_1_layer_call_fn_60308
5__inference_batch_normalization_1_layer_call_fn_60321┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
▓2»
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_58503Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ќ2ћ
/__inference_max_pooling2d_1_layer_call_fn_58509Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
я2█
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_60339
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_60357┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
5__inference_batch_normalization_2_layer_call_fn_60370
5__inference_batch_normalization_2_layer_call_fn_60383┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
▓2»
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_58641Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ќ2ћ
/__inference_max_pooling2d_2_layer_call_fn_58647Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
я2█
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_60401
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_60419┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
5__inference_batch_normalization_3_layer_call_fn_60432
5__inference_batch_normalization_3_layer_call_fn_60445┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
▓2»
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_58779Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ќ2ћ
/__inference_max_pooling2d_3_layer_call_fn_58785Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
║2и┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
║2и┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
я2█
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_60465
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_60499┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
5__inference_batch_normalization_4_layer_call_fn_60512
5__inference_batch_normalization_4_layer_call_fn_60525┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
║2и┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
║2и┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ш
 __inference__wrapped_model_56631л@ЏюЮъЪаАбБцЦдДеЕфФгГ«»░▒▓│┤иИХх╣║EбB
;б8
6і3
module_wrapper_input         dd
ф "EфB
@
module_wrapper_23+і(
module_wrapper_23         №
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_60277џБцЦдMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ №
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_60295џБцЦдMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ К
5__inference_batch_normalization_1_layer_call_fn_60308ЇБцЦдMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @К
5__inference_batch_normalization_1_layer_call_fn_60321ЇБцЦдMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @№
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_60339џЕфФгMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ №
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_60357џЕфФгMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ К
5__inference_batch_normalization_2_layer_call_fn_60370ЇЕфФгMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @К
5__inference_batch_normalization_2_layer_call_fn_60383ЇЕфФгMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_60401ю»░▒▓NбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_60419ю»░▒▓NбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ╔
5__inference_batch_normalization_3_layer_call_fn_60432Ј»░▒▓NбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђ╔
5__inference_batch_normalization_3_layer_call_fn_60445Ј»░▒▓NбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђ╝
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_60465hиИХх4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ ╝
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_60499hиИХх4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ћ
5__inference_batch_normalization_4_layer_call_fn_60512[иИХх4б1
*б'
!і
inputs         ђ
p 
ф "і         ђћ
5__inference_batch_normalization_4_layer_call_fn_60525[иИХх4б1
*б'
!і
inputs         ђ
p
ф "і         ђь
N__inference_batch_normalization_layer_call_and_return_conditional_losses_60215џЮъЪаMбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ ь
N__inference_batch_normalization_layer_call_and_return_conditional_losses_60233џЮъЪаMбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ┼
3__inference_batch_normalization_layer_call_fn_60246ЇЮъЪаMбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            ┼
3__inference_batch_normalization_layer_call_fn_60259ЇЮъЪаMбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            ь
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_58503ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ┼
/__inference_max_pooling2d_1_layer_call_fn_58509ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ь
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_58641ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ┼
/__inference_max_pooling2d_2_layer_call_fn_58647ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ь
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_58779ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ┼
/__inference_max_pooling2d_3_layer_call_fn_58785ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    в
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_58365ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ├
-__inference_max_pooling2d_layer_call_fn_58371ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    М
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_59721ѓЕфФгGбD
-б*
(і%
args_0         @
ф

trainingp "-б*
#і 
0         @
џ М
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_59739ѓЕфФгGбD
-б*
(і%
args_0         @
ф

trainingp"-б*
#і 
0         @
џ ф
1__inference_module_wrapper_10_layer_call_fn_59752uЕфФгGбD
-б*
(і%
args_0         @
ф

trainingp " і         @ф
1__inference_module_wrapper_10_layer_call_fn_59765uЕфФгGбD
-б*
(і%
args_0         @
ф

trainingp" і         @╚
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_59770xGбD
-б*
(і%
args_0         @
ф

trainingp "-б*
#і 
0         @
џ ╚
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_59775xGбD
-б*
(і%
args_0         @
ф

trainingp"-б*
#і 
0         @
џ а
1__inference_module_wrapper_11_layer_call_fn_59780kGбD
-б*
(і%
args_0         @
ф

trainingp " і         @а
1__inference_module_wrapper_11_layer_call_fn_59785kGбD
-б*
(і%
args_0         @
ф

trainingp" і         @¤
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_59795Г«GбD
-б*
(і%
args_0         @
ф

trainingp ".б+
$і!
0         ђ
џ ¤
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_59805Г«GбD
-б*
(і%
args_0         @
ф

trainingp".б+
$і!
0         ђ
џ Д
1__inference_module_wrapper_12_layer_call_fn_59814rГ«GбD
-б*
(і%
args_0         @
ф

trainingp "!і         ђД
1__inference_module_wrapper_12_layer_call_fn_59823rГ«GбD
-б*
(і%
args_0         @
ф

trainingp"!і         ђ╩
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_59828zHбE
.б+
)і&
args_0         ђ
ф

trainingp ".б+
$і!
0         ђ
џ ╩
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_59833zHбE
.б+
)і&
args_0         ђ
ф

trainingp".б+
$і!
0         ђ
џ б
1__inference_module_wrapper_13_layer_call_fn_59838mHбE
.б+
)і&
args_0         ђ
ф

trainingp "!і         ђб
1__inference_module_wrapper_13_layer_call_fn_59843mHбE
.б+
)і&
args_0         ђ
ф

trainingp"!і         ђН
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_59861ё»░▒▓HбE
.б+
)і&
args_0         ђ
ф

trainingp ".б+
$і!
0         ђ
џ Н
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_59879ё»░▒▓HбE
.б+
)і&
args_0         ђ
ф

trainingp".б+
$і!
0         ђ
џ г
1__inference_module_wrapper_14_layer_call_fn_59892w»░▒▓HбE
.б+
)і&
args_0         ђ
ф

trainingp "!і         ђг
1__inference_module_wrapper_14_layer_call_fn_59905w»░▒▓HбE
.б+
)і&
args_0         ђ
ф

trainingp"!і         ђ╩
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_59910zHбE
.б+
)і&
args_0         ђ
ф

trainingp ".б+
$і!
0         ђ
џ ╩
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_59915zHбE
.б+
)і&
args_0         ђ
ф

trainingp".б+
$і!
0         ђ
џ б
1__inference_module_wrapper_15_layer_call_fn_59920mHбE
.б+
)і&
args_0         ђ
ф

trainingp "!і         ђб
1__inference_module_wrapper_15_layer_call_fn_59925mHбE
.б+
)і&
args_0         ђ
ф

trainingp"!і         ђ╩
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_59930zHбE
.б+
)і&
args_0         ђ
ф

trainingp ".б+
$і!
0         ђ
џ ╩
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_59942zHбE
.б+
)і&
args_0         ђ
ф

trainingp".б+
$і!
0         ђ
џ б
1__inference_module_wrapper_16_layer_call_fn_59947mHбE
.б+
)і&
args_0         ђ
ф

trainingp "!і         ђб
1__inference_module_wrapper_16_layer_call_fn_59952mHбE
.б+
)і&
args_0         ђ
ф

trainingp"!і         ђ┬
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_59958rHбE
.б+
)і&
args_0         ђ
ф

trainingp "&б#
і
0         ђ
џ ┬
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_59964rHбE
.б+
)і&
args_0         ђ
ф

trainingp"&б#
і
0         ђ
џ џ
1__inference_module_wrapper_17_layer_call_fn_59969eHбE
.б+
)і&
args_0         ђ
ф

trainingp "і         ђџ
1__inference_module_wrapper_17_layer_call_fn_59974eHбE
.б+
)і&
args_0         ђ
ф

trainingp"і         ђ└
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_59984p│┤@б=
&б#
!і
args_0         ђ
ф

trainingp "&б#
і
0         ђ
џ └
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_59994p│┤@б=
&б#
!і
args_0         ђ
ф

trainingp"&б#
і
0         ђ
џ ў
1__inference_module_wrapper_18_layer_call_fn_60003c│┤@б=
&б#
!і
args_0         ђ
ф

trainingp "і         ђў
1__inference_module_wrapper_18_layer_call_fn_60012c│┤@б=
&б#
!і
args_0         ђ
ф

trainingp"і         ђ║
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_60017j@б=
&б#
!і
args_0         ђ
ф

trainingp "&б#
і
0         ђ
џ ║
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_60022j@б=
&б#
!і
args_0         ђ
ф

trainingp"&б#
і
0         ђ
џ њ
1__inference_module_wrapper_19_layer_call_fn_60027]@б=
&б#
!і
args_0         ђ
ф

trainingp "і         ђњ
1__inference_module_wrapper_19_layer_call_fn_60032]@б=
&б#
!і
args_0         ђ
ф

trainingp"і         ђК
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_59408xGбD
-б*
(і%
args_0         dd 
ф

trainingp "-б*
#і 
0         dd 
џ К
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_59413xGбD
-б*
(і%
args_0         dd 
ф

trainingp"-б*
#і 
0         dd 
џ Ъ
0__inference_module_wrapper_1_layer_call_fn_59418kGбD
-б*
(і%
args_0         dd 
ф

trainingp " і         dd Ъ
0__inference_module_wrapper_1_layer_call_fn_59423kGбD
-б*
(і%
args_0         dd 
ф

trainingp" і         dd ─
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_60052tиИХх@б=
&б#
!і
args_0         ђ
ф

trainingp "&б#
і
0         ђ
џ ─
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_60086tиИХх@б=
&б#
!і
args_0         ђ
ф

trainingp"&б#
і
0         ђ
џ ю
1__inference_module_wrapper_20_layer_call_fn_60099gиИХх@б=
&б#
!і
args_0         ђ
ф

trainingp "і         ђю
1__inference_module_wrapper_20_layer_call_fn_60112gиИХх@б=
&б#
!і
args_0         ђ
ф

trainingp"і         ђ║
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_60117j@б=
&б#
!і
args_0         ђ
ф

trainingp "&б#
і
0         ђ
џ ║
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_60129j@б=
&б#
!і
args_0         ђ
ф

trainingp"&б#
і
0         ђ
џ њ
1__inference_module_wrapper_21_layer_call_fn_60134]@б=
&б#
!і
args_0         ђ
ф

trainingp "і         ђњ
1__inference_module_wrapper_21_layer_call_fn_60139]@б=
&б#
!і
args_0         ђ
ф

trainingp"і         ђ┐
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_60149o╣║@б=
&б#
!і
args_0         ђ
ф

trainingp "%б"
і
0         
џ ┐
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_60159o╣║@б=
&б#
!і
args_0         ђ
ф

trainingp"%б"
і
0         
џ Ќ
1__inference_module_wrapper_22_layer_call_fn_60168b╣║@б=
&б#
!і
args_0         ђ
ф

trainingp "і         Ќ
1__inference_module_wrapper_22_layer_call_fn_60177b╣║@б=
&б#
!і
args_0         ђ
ф

trainingp"і         И
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_60182h?б<
%б"
 і
args_0         
ф

trainingp "%б"
і
0         
џ И
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_60187h?б<
%б"
 і
args_0         
ф

trainingp"%б"
і
0         
џ љ
1__inference_module_wrapper_23_layer_call_fn_60192[?б<
%б"
 і
args_0         
ф

trainingp "і         љ
1__inference_module_wrapper_23_layer_call_fn_60197[?б<
%б"
 і
args_0         
ф

trainingp"і         м
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_59441ѓЮъЪаGбD
-б*
(і%
args_0         dd 
ф

trainingp "-б*
#і 
0         dd 
џ м
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_59459ѓЮъЪаGбD
-б*
(і%
args_0         dd 
ф

trainingp"-б*
#і 
0         dd 
џ Е
0__inference_module_wrapper_2_layer_call_fn_59472uЮъЪаGбD
-б*
(і%
args_0         dd 
ф

trainingp " і         dd Е
0__inference_module_wrapper_2_layer_call_fn_59485uЮъЪаGбD
-б*
(і%
args_0         dd 
ф

trainingp" і         dd К
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_59490xGбD
-б*
(і%
args_0         dd 
ф

trainingp "-б*
#і 
0         !! 
џ К
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_59495xGбD
-б*
(і%
args_0         dd 
ф

trainingp"-б*
#і 
0         !! 
џ Ъ
0__inference_module_wrapper_3_layer_call_fn_59500kGбD
-б*
(і%
args_0         dd 
ф

trainingp " і         !! Ъ
0__inference_module_wrapper_3_layer_call_fn_59505kGбD
-б*
(і%
args_0         dd 
ф

trainingp" і         !! ═
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_59515~АбGбD
-б*
(і%
args_0         !! 
ф

trainingp "-б*
#і 
0         !!@
џ ═
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_59525~АбGбD
-б*
(і%
args_0         !! 
ф

trainingp"-б*
#і 
0         !!@
џ Ц
0__inference_module_wrapper_4_layer_call_fn_59534qАбGбD
-б*
(і%
args_0         !! 
ф

trainingp " і         !!@Ц
0__inference_module_wrapper_4_layer_call_fn_59543qАбGбD
-б*
(і%
args_0         !! 
ф

trainingp" і         !!@К
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_59548xGбD
-б*
(і%
args_0         !!@
ф

trainingp "-б*
#і 
0         !!@
џ К
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_59553xGбD
-б*
(і%
args_0         !!@
ф

trainingp"-б*
#і 
0         !!@
џ Ъ
0__inference_module_wrapper_5_layer_call_fn_59558kGбD
-б*
(і%
args_0         !!@
ф

trainingp " і         !!@Ъ
0__inference_module_wrapper_5_layer_call_fn_59563kGбD
-б*
(і%
args_0         !!@
ф

trainingp" і         !!@м
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_59581ѓБцЦдGбD
-б*
(і%
args_0         !!@
ф

trainingp "-б*
#і 
0         !!@
џ м
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_59599ѓБцЦдGбD
-б*
(і%
args_0         !!@
ф

trainingp"-б*
#і 
0         !!@
џ Е
0__inference_module_wrapper_6_layer_call_fn_59612uБцЦдGбD
-б*
(і%
args_0         !!@
ф

trainingp " і         !!@Е
0__inference_module_wrapper_6_layer_call_fn_59625uБцЦдGбD
-б*
(і%
args_0         !!@
ф

trainingp" і         !!@К
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_59630xGбD
-б*
(і%
args_0         !!@
ф

trainingp "-б*
#і 
0         @
џ К
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_59635xGбD
-б*
(і%
args_0         !!@
ф

trainingp"-б*
#і 
0         @
џ Ъ
0__inference_module_wrapper_7_layer_call_fn_59640kGбD
-б*
(і%
args_0         !!@
ф

trainingp " і         @Ъ
0__inference_module_wrapper_7_layer_call_fn_59645kGбD
-б*
(і%
args_0         !!@
ф

trainingp" і         @═
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_59655~ДеGбD
-б*
(і%
args_0         @
ф

trainingp "-б*
#і 
0         @
џ ═
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_59665~ДеGбD
-б*
(і%
args_0         @
ф

trainingp"-б*
#і 
0         @
џ Ц
0__inference_module_wrapper_8_layer_call_fn_59674qДеGбD
-б*
(і%
args_0         @
ф

trainingp " і         @Ц
0__inference_module_wrapper_8_layer_call_fn_59683qДеGбD
-б*
(і%
args_0         @
ф

trainingp" і         @К
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_59688xGбD
-б*
(і%
args_0         @
ф

trainingp "-б*
#і 
0         @
џ К
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_59693xGбD
-б*
(і%
args_0         @
ф

trainingp"-б*
#і 
0         @
џ Ъ
0__inference_module_wrapper_9_layer_call_fn_59698kGбD
-б*
(і%
args_0         @
ф

trainingp " і         @Ъ
0__inference_module_wrapper_9_layer_call_fn_59703kGбD
-б*
(і%
args_0         @
ф

trainingp" і         @╦
I__inference_module_wrapper_layer_call_and_return_conditional_losses_59375~ЏюGбD
-б*
(і%
args_0         dd
ф

trainingp "-б*
#і 
0         dd 
џ ╦
I__inference_module_wrapper_layer_call_and_return_conditional_losses_59385~ЏюGбD
-б*
(і%
args_0         dd
ф

trainingp"-б*
#і 
0         dd 
џ Б
.__inference_module_wrapper_layer_call_fn_59394qЏюGбD
-б*
(і%
args_0         dd
ф

trainingp " і         dd Б
.__inference_module_wrapper_layer_call_fn_59403qЏюGбD
-б*
(і%
args_0         dd
ф

trainingp" і         dd ѓ
E__inference_sequential_layer_call_and_return_conditional_losses_58065И@ЏюЮъЪаАбБцЦдДеЕфФгГ«»░▒▓│┤иИХх╣║MбJ
Cб@
6і3
module_wrapper_input         dd
p 

 
ф "%б"
і
0         
џ ѓ
E__inference_sequential_layer_call_and_return_conditional_losses_58157И@ЏюЮъЪаАбБцЦдДеЕфФгГ«»░▒▓│┤иИХх╣║MбJ
Cб@
6і3
module_wrapper_input         dd
p

 
ф "%б"
і
0         
џ З
E__inference_sequential_layer_call_and_return_conditional_losses_59073ф@ЏюЮъЪаАбБцЦдДеЕфФгГ«»░▒▓│┤иИХх╣║?б<
5б2
(і%
inputs         dd
p 

 
ф "%б"
і
0         
џ З
E__inference_sequential_layer_call_and_return_conditional_losses_59227ф@ЏюЮъЪаАбБцЦдДеЕфФгГ«»░▒▓│┤иИХх╣║?б<
5б2
(і%
inputs         dd
p

 
ф "%б"
і
0         
џ ┌
*__inference_sequential_layer_call_fn_57036Ф@ЏюЮъЪаАбБцЦдДеЕфФгГ«»░▒▓│┤иИХх╣║MбJ
Cб@
6і3
module_wrapper_input         dd
p 

 
ф "і         ┌
*__inference_sequential_layer_call_fn_57973Ф@ЏюЮъЪаАбБцЦдДеЕфФгГ«»░▒▓│┤иИХх╣║MбJ
Cб@
6і3
module_wrapper_input         dd
p

 
ф "і         ╠
*__inference_sequential_layer_call_fn_59296Ю@ЏюЮъЪаАбБцЦдДеЕфФгГ«»░▒▓│┤иИХх╣║?б<
5б2
(і%
inputs         dd
p 

 
ф "і         ╠
*__inference_sequential_layer_call_fn_59365Ю@ЏюЮъЪаАбБцЦдДеЕфФгГ«»░▒▓│┤иИХх╣║?б<
5б2
(і%
inputs         dd
p

 
ф "і         љ
#__inference_signature_wrapper_58232У@ЏюЮъЪаАбБцЦдДеЕфФгГ«»░▒▓│┤иИХх╣║]бZ
б 
SфP
N
module_wrapper_input6і3
module_wrapper_input         dd"EфB
@
module_wrapper_23+і(
module_wrapper_23         