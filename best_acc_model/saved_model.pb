ЭЩ)
╓ж
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
Ы
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
;
Elu
features"T
activations"T"
Ttype:
2
·
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
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
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
В
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
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
list(type)(0И
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
list(type)(0И
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
╛
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
executor_typestring И
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ул!
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
Ь
module_wrapper/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namemodule_wrapper/conv2d/kernel
Х
0module_wrapper/conv2d/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper/conv2d/kernel*&
_output_shapes
: *
dtype0
М
module_wrapper/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namemodule_wrapper/conv2d/bias
Е
.module_wrapper/conv2d/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper/conv2d/bias*
_output_shapes
: *
dtype0
м
*module_wrapper_2/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*module_wrapper_2/batch_normalization/gamma
е
>module_wrapper_2/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp*module_wrapper_2/batch_normalization/gamma*
_output_shapes
: *
dtype0
к
)module_wrapper_2/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)module_wrapper_2/batch_normalization/beta
г
=module_wrapper_2/batch_normalization/beta/Read/ReadVariableOpReadVariableOp)module_wrapper_2/batch_normalization/beta*
_output_shapes
: *
dtype0
д
 module_wrapper_4/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*1
shared_name" module_wrapper_4/conv2d_1/kernel
Э
4module_wrapper_4/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_4/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
Ф
module_wrapper_4/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name module_wrapper_4/conv2d_1/bias
Н
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
й
@module_wrapper_6/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp,module_wrapper_6/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
о
+module_wrapper_6/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+module_wrapper_6/batch_normalization_1/beta
з
?module_wrapper_6/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp+module_wrapper_6/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
д
 module_wrapper_8/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*1
shared_name" module_wrapper_8/conv2d_2/kernel
Э
4module_wrapper_8/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_8/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0
Ф
module_wrapper_8/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name module_wrapper_8/conv2d_2/bias
Н
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
л
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
й
@module_wrapper_10/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp,module_wrapper_10/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
з
!module_wrapper_12/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*2
shared_name#!module_wrapper_12/conv2d_3/kernel
а
5module_wrapper_12/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp!module_wrapper_12/conv2d_3/kernel*'
_output_shapes
:@А*
dtype0
Ч
module_wrapper_12/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!module_wrapper_12/conv2d_3/bias
Р
3module_wrapper_12/conv2d_3/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_12/conv2d_3/bias*
_output_shapes	
:А*
dtype0
│
-module_wrapper_14/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*>
shared_name/-module_wrapper_14/batch_normalization_3/gamma
м
Amodule_wrapper_14/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp-module_wrapper_14/batch_normalization_3/gamma*
_output_shapes	
:А*
dtype0
▒
,module_wrapper_14/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*=
shared_name.,module_wrapper_14/batch_normalization_3/beta
к
@module_wrapper_14/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp,module_wrapper_14/batch_normalization_3/beta*
_output_shapes	
:А*
dtype0
Ъ
module_wrapper_18/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*/
shared_name module_wrapper_18/dense/kernel
У
2module_wrapper_18/dense/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper_18/dense/kernel* 
_output_shapes
:
АА*
dtype0
С
module_wrapper_18/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namemodule_wrapper_18/dense/bias
К
0module_wrapper_18/dense/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_18/dense/bias*
_output_shapes	
:А*
dtype0
│
-module_wrapper_20/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*>
shared_name/-module_wrapper_20/batch_normalization_4/gamma
м
Amodule_wrapper_20/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp-module_wrapper_20/batch_normalization_4/gamma*
_output_shapes	
:А*
dtype0
▒
,module_wrapper_20/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*=
shared_name.,module_wrapper_20/batch_normalization_4/beta
к
@module_wrapper_20/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp,module_wrapper_20/batch_normalization_4/beta*
_output_shapes	
:А*
dtype0
Э
 module_wrapper_22/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*1
shared_name" module_wrapper_22/dense_1/kernel
Ц
4module_wrapper_22/dense_1/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_22/dense_1/kernel*
_output_shapes
:	А*
dtype0
Ф
module_wrapper_22/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name module_wrapper_22/dense_1/bias
Н
2module_wrapper_22/dense_1/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_22/dense_1/bias*
_output_shapes
:*
dtype0
╕
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
╝
2module_wrapper_6/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42module_wrapper_6/batch_normalization_1/moving_mean
╡
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
╜
Jmodule_wrapper_6/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp6module_wrapper_6/batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
╛
3module_wrapper_10/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53module_wrapper_10/batch_normalization_2/moving_mean
╖
Gmodule_wrapper_10/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp3module_wrapper_10/batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
╞
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
┐
3module_wrapper_14/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*D
shared_name53module_wrapper_14/batch_normalization_3/moving_mean
╕
Gmodule_wrapper_14/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp3module_wrapper_14/batch_normalization_3/moving_mean*
_output_shapes	
:А*
dtype0
╟
7module_wrapper_14/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*H
shared_name97module_wrapper_14/batch_normalization_3/moving_variance
└
Kmodule_wrapper_14/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp7module_wrapper_14/batch_normalization_3/moving_variance*
_output_shapes	
:А*
dtype0
┐
3module_wrapper_20/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*D
shared_name53module_wrapper_20/batch_normalization_4/moving_mean
╕
Gmodule_wrapper_20/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp3module_wrapper_20/batch_normalization_4/moving_mean*
_output_shapes	
:А*
dtype0
╟
7module_wrapper_20/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*H
shared_name97module_wrapper_20/batch_normalization_4/moving_variance
└
Kmodule_wrapper_20/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp7module_wrapper_20/batch_normalization_4/moving_variance*
_output_shapes	
:А*
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
╢
)SGD/module_wrapper/conv2d/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)SGD/module_wrapper/conv2d/kernel/momentum
п
=SGD/module_wrapper/conv2d/kernel/momentum/Read/ReadVariableOpReadVariableOp)SGD/module_wrapper/conv2d/kernel/momentum*&
_output_shapes
: *
dtype0
ж
'SGD/module_wrapper/conv2d/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/module_wrapper/conv2d/bias/momentum
Я
;SGD/module_wrapper/conv2d/bias/momentum/Read/ReadVariableOpReadVariableOp'SGD/module_wrapper/conv2d/bias/momentum*
_output_shapes
: *
dtype0
╞
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
╜
JSGD/module_wrapper_2/batch_normalization/beta/momentum/Read/ReadVariableOpReadVariableOp6SGD/module_wrapper_2/batch_normalization/beta/momentum*
_output_shapes
: *
dtype0
╛
-SGD/module_wrapper_4/conv2d_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*>
shared_name/-SGD/module_wrapper_4/conv2d_1/kernel/momentum
╖
ASGD/module_wrapper_4/conv2d_1/kernel/momentum/Read/ReadVariableOpReadVariableOp-SGD/module_wrapper_4/conv2d_1/kernel/momentum*&
_output_shapes
: @*
dtype0
о
+SGD/module_wrapper_4/conv2d_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+SGD/module_wrapper_4/conv2d_1/bias/momentum
з
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
╛
-SGD/module_wrapper_8/conv2d_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*>
shared_name/-SGD/module_wrapper_8/conv2d_2/kernel/momentum
╖
ASGD/module_wrapper_8/conv2d_2/kernel/momentum/Read/ReadVariableOpReadVariableOp-SGD/module_wrapper_8/conv2d_2/kernel/momentum*&
_output_shapes
:@@*
dtype0
о
+SGD/module_wrapper_8/conv2d_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+SGD/module_wrapper_8/conv2d_2/bias/momentum
з
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
shape:@А*?
shared_name0.SGD/module_wrapper_12/conv2d_3/kernel/momentum
║
BSGD/module_wrapper_12/conv2d_3/kernel/momentum/Read/ReadVariableOpReadVariableOp.SGD/module_wrapper_12/conv2d_3/kernel/momentum*'
_output_shapes
:@А*
dtype0
▒
,SGD/module_wrapper_12/conv2d_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*=
shared_name.,SGD/module_wrapper_12/conv2d_3/bias/momentum
к
@SGD/module_wrapper_12/conv2d_3/bias/momentum/Read/ReadVariableOpReadVariableOp,SGD/module_wrapper_12/conv2d_3/bias/momentum*
_output_shapes	
:А*
dtype0
═
:SGD/module_wrapper_14/batch_normalization_3/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*K
shared_name<:SGD/module_wrapper_14/batch_normalization_3/gamma/momentum
╞
NSGD/module_wrapper_14/batch_normalization_3/gamma/momentum/Read/ReadVariableOpReadVariableOp:SGD/module_wrapper_14/batch_normalization_3/gamma/momentum*
_output_shapes	
:А*
dtype0
╦
9SGD/module_wrapper_14/batch_normalization_3/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*J
shared_name;9SGD/module_wrapper_14/batch_normalization_3/beta/momentum
─
MSGD/module_wrapper_14/batch_normalization_3/beta/momentum/Read/ReadVariableOpReadVariableOp9SGD/module_wrapper_14/batch_normalization_3/beta/momentum*
_output_shapes	
:А*
dtype0
┤
+SGD/module_wrapper_18/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*<
shared_name-+SGD/module_wrapper_18/dense/kernel/momentum
н
?SGD/module_wrapper_18/dense/kernel/momentum/Read/ReadVariableOpReadVariableOp+SGD/module_wrapper_18/dense/kernel/momentum* 
_output_shapes
:
АА*
dtype0
л
)SGD/module_wrapper_18/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*:
shared_name+)SGD/module_wrapper_18/dense/bias/momentum
д
=SGD/module_wrapper_18/dense/bias/momentum/Read/ReadVariableOpReadVariableOp)SGD/module_wrapper_18/dense/bias/momentum*
_output_shapes	
:А*
dtype0
═
:SGD/module_wrapper_20/batch_normalization_4/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*K
shared_name<:SGD/module_wrapper_20/batch_normalization_4/gamma/momentum
╞
NSGD/module_wrapper_20/batch_normalization_4/gamma/momentum/Read/ReadVariableOpReadVariableOp:SGD/module_wrapper_20/batch_normalization_4/gamma/momentum*
_output_shapes	
:А*
dtype0
╦
9SGD/module_wrapper_20/batch_normalization_4/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*J
shared_name;9SGD/module_wrapper_20/batch_normalization_4/beta/momentum
─
MSGD/module_wrapper_20/batch_normalization_4/beta/momentum/Read/ReadVariableOpReadVariableOp9SGD/module_wrapper_20/batch_normalization_4/beta/momentum*
_output_shapes	
:А*
dtype0
╖
-SGD/module_wrapper_22/dense_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*>
shared_name/-SGD/module_wrapper_22/dense_1/kernel/momentum
░
ASGD/module_wrapper_22/dense_1/kernel/momentum/Read/ReadVariableOpReadVariableOp-SGD/module_wrapper_22/dense_1/kernel/momentum*
_output_shapes
:	А*
dtype0
о
+SGD/module_wrapper_22/dense_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+SGD/module_wrapper_22/dense_1/bias/momentum
з
?SGD/module_wrapper_22/dense_1/bias/momentum/Read/ReadVariableOpReadVariableOp+SGD/module_wrapper_22/dense_1/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp
╬╚
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*И╚
value¤╟B∙╟ Bё╟
╓
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
_
_module
 trainable_variables
!regularization_losses
"	variables
#	keras_api
_
$_module
%trainable_variables
&regularization_losses
'	variables
(	keras_api
_
)_module
*trainable_variables
+regularization_losses
,	variables
-	keras_api
_
._module
/trainable_variables
0regularization_losses
1	variables
2	keras_api
_
3_module
4trainable_variables
5regularization_losses
6	variables
7	keras_api
_
8_module
9trainable_variables
:regularization_losses
;	variables
<	keras_api
_
=_module
>trainable_variables
?regularization_losses
@	variables
A	keras_api
_
B_module
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
_
G_module
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
_
L_module
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
_
Q_module
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
_
V_module
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
_
[_module
\trainable_variables
]regularization_losses
^	variables
_	keras_api
_
`_module
atrainable_variables
bregularization_losses
c	variables
d	keras_api
_
e_module
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
_
j_module
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
_
o_module
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
_
t_module
utrainable_variables
vregularization_losses
w	variables
x	keras_api
_
y_module
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
b
~_module
trainable_variables
Аregularization_losses
Б	variables
В	keras_api
d
Г_module
Дtrainable_variables
Еregularization_losses
Ж	variables
З	keras_api
d
И_module
Йtrainable_variables
Кregularization_losses
Л	variables
М	keras_api
d
Н_module
Оtrainable_variables
Пregularization_losses
Р	variables
С	keras_api
d
Т_module
Уtrainable_variables
Фregularization_losses
Х	variables
Ц	keras_api
╞
	Чiter

Шdecay
Щlearning_rate
ЪmomentumЫmomentumаЬmomentumбЭmomentumвЮmomentumгЯmomentumдаmomentumебmomentumжвmomentumзгmomentumидmomentumйеmomentumкжmomentumлзmomentumмиmomentumнйmomentumокmomentumплmomentum░мmomentum▒нmomentum▓оmomentum│пmomentum┤░momentum╡
╝
Ы0
Ь1
Э2
Ю3
Я4
а5
б6
в7
г8
д9
е10
ж11
з12
и13
й14
к15
л16
м17
н18
о19
п20
░21
 
Ц
Ы0
Ь1
Э2
Ю3
▒4
▓5
Я6
а7
б8
в9
│10
┤11
г12
д13
е14
ж15
╡16
╢17
з18
и19
й20
к21
╖22
╕23
л24
м25
н26
о27
╣28
║29
п30
░31
▓
╗metrics
trainable_variables
regularization_losses
 ╝layer_regularization_losses
	variables
╜non_trainable_variables
╛layer_metrics
┐layers
 
n
Ыkernel
	Ьbias
└trainable_variables
┴regularization_losses
┬	variables
├	keras_api

Ы0
Ь1
 

Ы0
Ь1
▓
─metrics
 trainable_variables
!regularization_losses
 ┼layer_regularization_losses
"	variables
╞non_trainable_variables
╟layer_metrics
╚layers
V
╔trainable_variables
╩regularization_losses
╦	variables
╠	keras_api
 
 
 
▓
═metrics
%trainable_variables
&regularization_losses
 ╬layer_regularization_losses
'	variables
╧non_trainable_variables
╨layer_metrics
╤layers
а
	╥axis

Эgamma
	Юbeta
▒moving_mean
▓moving_variance
╙trainable_variables
╘regularization_losses
╒	variables
╓	keras_api

Э0
Ю1
 
 
Э0
Ю1
▒2
▓3
▓
╫metrics
*trainable_variables
+regularization_losses
 ╪layer_regularization_losses
,	variables
┘non_trainable_variables
┌layer_metrics
█layers
V
▄trainable_variables
▌regularization_losses
▐	variables
▀	keras_api
 
 
 
▓
рmetrics
/trainable_variables
0regularization_losses
 сlayer_regularization_losses
1	variables
тnon_trainable_variables
уlayer_metrics
фlayers
n
Яkernel
	аbias
хtrainable_variables
цregularization_losses
ч	variables
ш	keras_api

Я0
а1
 

Я0
а1
▓
щmetrics
4trainable_variables
5regularization_losses
 ъlayer_regularization_losses
6	variables
ыnon_trainable_variables
ьlayer_metrics
эlayers
V
юtrainable_variables
яregularization_losses
Ё	variables
ё	keras_api
 
 
 
▓
Єmetrics
9trainable_variables
:regularization_losses
 єlayer_regularization_losses
;	variables
Їnon_trainable_variables
їlayer_metrics
Ўlayers
а
	ўaxis

бgamma
	вbeta
│moving_mean
┤moving_variance
°trainable_variables
∙regularization_losses
·	variables
√	keras_api

б0
в1
 
 
б0
в1
│2
┤3
▓
№metrics
>trainable_variables
?regularization_losses
 ¤layer_regularization_losses
@	variables
■non_trainable_variables
 layer_metrics
Аlayers
V
Бtrainable_variables
Вregularization_losses
Г	variables
Д	keras_api
 
 
 
▓
Еmetrics
Ctrainable_variables
Dregularization_losses
 Жlayer_regularization_losses
E	variables
Зnon_trainable_variables
Иlayer_metrics
Йlayers
n
гkernel
	дbias
Кtrainable_variables
Лregularization_losses
М	variables
Н	keras_api

г0
д1
 

г0
д1
▓
Оmetrics
Htrainable_variables
Iregularization_losses
 Пlayer_regularization_losses
J	variables
Рnon_trainable_variables
Сlayer_metrics
Тlayers
V
Уtrainable_variables
Фregularization_losses
Х	variables
Ц	keras_api
 
 
 
▓
Чmetrics
Mtrainable_variables
Nregularization_losses
 Шlayer_regularization_losses
O	variables
Щnon_trainable_variables
Ъlayer_metrics
Ыlayers
а
	Ьaxis

еgamma
	жbeta
╡moving_mean
╢moving_variance
Эtrainable_variables
Юregularization_losses
Я	variables
а	keras_api

е0
ж1
 
 
е0
ж1
╡2
╢3
▓
бmetrics
Rtrainable_variables
Sregularization_losses
 вlayer_regularization_losses
T	variables
гnon_trainable_variables
дlayer_metrics
еlayers
V
жtrainable_variables
зregularization_losses
и	variables
й	keras_api
 
 
 
▓
кmetrics
Wtrainable_variables
Xregularization_losses
 лlayer_regularization_losses
Y	variables
мnon_trainable_variables
нlayer_metrics
оlayers
n
зkernel
	иbias
пtrainable_variables
░regularization_losses
▒	variables
▓	keras_api

з0
и1
 

з0
и1
▓
│metrics
\trainable_variables
]regularization_losses
 ┤layer_regularization_losses
^	variables
╡non_trainable_variables
╢layer_metrics
╖layers
V
╕trainable_variables
╣regularization_losses
║	variables
╗	keras_api
 
 
 
▓
╝metrics
atrainable_variables
bregularization_losses
 ╜layer_regularization_losses
c	variables
╛non_trainable_variables
┐layer_metrics
└layers
а
	┴axis

йgamma
	кbeta
╖moving_mean
╕moving_variance
┬trainable_variables
├regularization_losses
─	variables
┼	keras_api

й0
к1
 
 
й0
к1
╖2
╕3
▓
╞metrics
ftrainable_variables
gregularization_losses
 ╟layer_regularization_losses
h	variables
╚non_trainable_variables
╔layer_metrics
╩layers
V
╦trainable_variables
╠regularization_losses
═	variables
╬	keras_api
 
 
 
▓
╧metrics
ktrainable_variables
lregularization_losses
 ╨layer_regularization_losses
m	variables
╤non_trainable_variables
╥layer_metrics
╙layers
V
╘trainable_variables
╒regularization_losses
╓	variables
╫	keras_api
 
 
 
▓
╪metrics
ptrainable_variables
qregularization_losses
 ┘layer_regularization_losses
r	variables
┌non_trainable_variables
█layer_metrics
▄layers
V
▌trainable_variables
▐regularization_losses
▀	variables
р	keras_api
 
 
 
▓
сmetrics
utrainable_variables
vregularization_losses
 тlayer_regularization_losses
w	variables
уnon_trainable_variables
фlayer_metrics
хlayers
n
лkernel
	мbias
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api

л0
м1
 

л0
м1
▓
ъmetrics
ztrainable_variables
{regularization_losses
 ыlayer_regularization_losses
|	variables
ьnon_trainable_variables
эlayer_metrics
юlayers
V
яtrainable_variables
Ёregularization_losses
ё	variables
Є	keras_api
 
 
 
┤
єmetrics
trainable_variables
Аregularization_losses
 Їlayer_regularization_losses
Б	variables
їnon_trainable_variables
Ўlayer_metrics
ўlayers
а
	°axis

нgamma
	оbeta
╣moving_mean
║moving_variance
∙trainable_variables
·regularization_losses
√	variables
№	keras_api

н0
о1
 
 
н0
о1
╣2
║3
╡
¤metrics
Дtrainable_variables
Еregularization_losses
 ■layer_regularization_losses
Ж	variables
 non_trainable_variables
Аlayer_metrics
Бlayers
V
Вtrainable_variables
Гregularization_losses
Д	variables
Е	keras_api
 
 
 
╡
Жmetrics
Йtrainable_variables
Кregularization_losses
 Зlayer_regularization_losses
Л	variables
Иnon_trainable_variables
Йlayer_metrics
Кlayers
n
пkernel
	░bias
Лtrainable_variables
Мregularization_losses
Н	variables
О	keras_api

п0
░1
 

п0
░1
╡
Пmetrics
Оtrainable_variables
Пregularization_losses
 Рlayer_regularization_losses
Р	variables
Сnon_trainable_variables
Тlayer_metrics
Уlayers
V
Фtrainable_variables
Хregularization_losses
Ц	variables
Ч	keras_api
 
 
 
╡
Шmetrics
Уtrainable_variables
Фregularization_losses
 Щlayer_regularization_losses
Х	variables
Ъnon_trainable_variables
Ыlayer_metrics
Ьlayers
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEmodule_wrapper/conv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEmodule_wrapper/conv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE*module_wrapper_2/batch_normalization/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE)module_wrapper_2/batch_normalization/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE module_wrapper_4/conv2d_1/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmodule_wrapper_4/conv2d_1/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE,module_wrapper_6/batch_normalization_1/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+module_wrapper_6/batch_normalization_1/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE module_wrapper_8/conv2d_2/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmodule_wrapper_8/conv2d_2/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-module_wrapper_10/batch_normalization_2/gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,module_wrapper_10/batch_normalization_2/beta1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!module_wrapper_12/conv2d_3/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmodule_wrapper_12/conv2d_3/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-module_wrapper_14/batch_normalization_3/gamma1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,module_wrapper_14/batch_normalization_3/beta1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEmodule_wrapper_18/dense/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEmodule_wrapper_18/dense/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-module_wrapper_20/batch_normalization_4/gamma1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,module_wrapper_20/batch_normalization_4/beta1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE module_wrapper_22/dense_1/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEmodule_wrapper_22/dense_1/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0module_wrapper_2/batch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE4module_wrapper_2/batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2module_wrapper_6/batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6module_wrapper_6/batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3module_wrapper_10/batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7module_wrapper_10/batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3module_wrapper_14/batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7module_wrapper_14/batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3module_wrapper_20/batch_normalization_4/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7module_wrapper_20/batch_normalization_4/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE

Э0
Ю1
 
P
▒0
▓1
│2
┤3
╡4
╢5
╖6
╕7
╣8
║9
 
╢
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
Ы0
Ь1
 

Ы0
Ь1
╡
Яmetrics
└trainable_variables
┴regularization_losses
 аlayer_regularization_losses
┬	variables
бnon_trainable_variables
вlayer_metrics
гlayers
 
 
 
 
 
 
 
 
╡
дmetrics
╔trainable_variables
╩regularization_losses
 еlayer_regularization_losses
╦	variables
жnon_trainable_variables
зlayer_metrics
иlayers
 
 
 
 
 
 

Э0
Ю1
 
 
Э0
Ю1
▒2
▓3
╡
йmetrics
╙trainable_variables
╘regularization_losses
 кlayer_regularization_losses
╒	variables
лnon_trainable_variables
мlayer_metrics
нlayers
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
╡
оmetrics
▄trainable_variables
▌regularization_losses
 пlayer_regularization_losses
▐	variables
░non_trainable_variables
▒layer_metrics
▓layers
 
 
 
 
 

Я0
а1
 

Я0
а1
╡
│metrics
хtrainable_variables
цregularization_losses
 ┤layer_regularization_losses
ч	variables
╡non_trainable_variables
╢layer_metrics
╖layers
 
 
 
 
 
 
 
 
╡
╕metrics
юtrainable_variables
яregularization_losses
 ╣layer_regularization_losses
Ё	variables
║non_trainable_variables
╗layer_metrics
╝layers
 
 
 
 
 
 

б0
в1
 
 
б0
в1
│2
┤3
╡
╜metrics
°trainable_variables
∙regularization_losses
 ╛layer_regularization_losses
·	variables
┐non_trainable_variables
└layer_metrics
┴layers
 
 

│0
┤1
 
 
 
 
 
╡
┬metrics
Бtrainable_variables
Вregularization_losses
 ├layer_regularization_losses
Г	variables
─non_trainable_variables
┼layer_metrics
╞layers
 
 
 
 
 

г0
д1
 

г0
д1
╡
╟metrics
Кtrainable_variables
Лregularization_losses
 ╚layer_regularization_losses
М	variables
╔non_trainable_variables
╩layer_metrics
╦layers
 
 
 
 
 
 
 
 
╡
╠metrics
Уtrainable_variables
Фregularization_losses
 ═layer_regularization_losses
Х	variables
╬non_trainable_variables
╧layer_metrics
╨layers
 
 
 
 
 
 

е0
ж1
 
 
е0
ж1
╡2
╢3
╡
╤metrics
Эtrainable_variables
Юregularization_losses
 ╥layer_regularization_losses
Я	variables
╙non_trainable_variables
╘layer_metrics
╒layers
 
 

╡0
╢1
 
 
 
 
 
╡
╓metrics
жtrainable_variables
зregularization_losses
 ╫layer_regularization_losses
и	variables
╪non_trainable_variables
┘layer_metrics
┌layers
 
 
 
 
 

з0
и1
 

з0
и1
╡
█metrics
пtrainable_variables
░regularization_losses
 ▄layer_regularization_losses
▒	variables
▌non_trainable_variables
▐layer_metrics
▀layers
 
 
 
 
 
 
 
 
╡
рmetrics
╕trainable_variables
╣regularization_losses
 сlayer_regularization_losses
║	variables
тnon_trainable_variables
уlayer_metrics
фlayers
 
 
 
 
 
 

й0
к1
 
 
й0
к1
╖2
╕3
╡
хmetrics
┬trainable_variables
├regularization_losses
 цlayer_regularization_losses
─	variables
чnon_trainable_variables
шlayer_metrics
щlayers
 
 

╖0
╕1
 
 
 
 
 
╡
ъmetrics
╦trainable_variables
╠regularization_losses
 ыlayer_regularization_losses
═	variables
ьnon_trainable_variables
эlayer_metrics
юlayers
 
 
 
 
 
 
 
 
╡
яmetrics
╘trainable_variables
╒regularization_losses
 Ёlayer_regularization_losses
╓	variables
ёnon_trainable_variables
Єlayer_metrics
єlayers
 
 
 
 
 
 
 
 
╡
Їmetrics
▌trainable_variables
▐regularization_losses
 їlayer_regularization_losses
▀	variables
Ўnon_trainable_variables
ўlayer_metrics
°layers
 
 
 
 
 

л0
м1
 

л0
м1
╡
∙metrics
цtrainable_variables
чregularization_losses
 ·layer_regularization_losses
ш	variables
√non_trainable_variables
№layer_metrics
¤layers
 
 
 
 
 
 
 
 
╡
■metrics
яtrainable_variables
Ёregularization_losses
  layer_regularization_losses
ё	variables
Аnon_trainable_variables
Бlayer_metrics
Вlayers
 
 
 
 
 
 

н0
о1
 
 
н0
о1
╣2
║3
╡
Гmetrics
∙trainable_variables
·regularization_losses
 Дlayer_regularization_losses
√	variables
Еnon_trainable_variables
Жlayer_metrics
Зlayers
 
 

╣0
║1
 
 
 
 
 
╡
Иmetrics
Вtrainable_variables
Гregularization_losses
 Йlayer_regularization_losses
Д	variables
Кnon_trainable_variables
Лlayer_metrics
Мlayers
 
 
 
 
 

п0
░1
 

п0
░1
╡
Нmetrics
Лtrainable_variables
Мregularization_losses
 Оlayer_regularization_losses
Н	variables
Пnon_trainable_variables
Рlayer_metrics
Сlayers
 
 
 
 
 
 
 
 
╡
Тmetrics
Фtrainable_variables
Хregularization_losses
 Уlayer_regularization_losses
Ц	variables
Фnon_trainable_variables
Хlayer_metrics
Цlayers
 
 
 
 
 
8

Чtotal

Шcount
Щ	variables
Ъ	keras_api
I

Ыtotal

Ьcount
Э
_fn_kwargs
Ю	variables
Я	keras_api
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

│0
┤1
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
╡0
╢1
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
╖0
╕1
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
╣0
║1
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
Ч0
Ш1

Щ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ы0
Ь1

Ю	variables
УР
VARIABLE_VALUE)SGD/module_wrapper/conv2d/kernel/momentumStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE'SGD/module_wrapper/conv2d/bias/momentumStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
бЮ
VARIABLE_VALUE7SGD/module_wrapper_2/batch_normalization/gamma/momentumStrainable_variables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
аЭ
VARIABLE_VALUE6SGD/module_wrapper_2/batch_normalization/beta/momentumStrainable_variables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE-SGD/module_wrapper_4/conv2d_1/kernel/momentumStrainable_variables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE+SGD/module_wrapper_4/conv2d_1/bias/momentumStrainable_variables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
га
VARIABLE_VALUE9SGD/module_wrapper_6/batch_normalization_1/gamma/momentumStrainable_variables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
вЯ
VARIABLE_VALUE8SGD/module_wrapper_6/batch_normalization_1/beta/momentumStrainable_variables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE-SGD/module_wrapper_8/conv2d_2/kernel/momentumStrainable_variables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE+SGD/module_wrapper_8/conv2d_2/bias/momentumStrainable_variables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ев
VARIABLE_VALUE:SGD/module_wrapper_10/batch_normalization_2/gamma/momentumTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
дб
VARIABLE_VALUE9SGD/module_wrapper_10/batch_normalization_2/beta/momentumTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE.SGD/module_wrapper_12/conv2d_3/kernel/momentumTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE,SGD/module_wrapper_12/conv2d_3/bias/momentumTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ев
VARIABLE_VALUE:SGD/module_wrapper_14/batch_normalization_3/gamma/momentumTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
дб
VARIABLE_VALUE9SGD/module_wrapper_14/batch_normalization_3/beta/momentumTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE+SGD/module_wrapper_18/dense/kernel/momentumTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE)SGD/module_wrapper_18/dense/bias/momentumTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ев
VARIABLE_VALUE:SGD/module_wrapper_20/batch_normalization_4/gamma/momentumTtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
дб
VARIABLE_VALUE9SGD/module_wrapper_20/batch_normalization_4/beta/momentumTtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE-SGD/module_wrapper_22/dense_1/kernel/momentumTtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE+SGD/module_wrapper_22/dense_1/bias/momentumTtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Ч
$serving_default_module_wrapper_inputPlaceholder*/
_output_shapes
:         dd*
dtype0*$
shape:         dd
ч
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
GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_96241
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╪ 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp0module_wrapper/conv2d/kernel/Read/ReadVariableOp.module_wrapper/conv2d/bias/Read/ReadVariableOp>module_wrapper_2/batch_normalization/gamma/Read/ReadVariableOp=module_wrapper_2/batch_normalization/beta/Read/ReadVariableOp4module_wrapper_4/conv2d_1/kernel/Read/ReadVariableOp2module_wrapper_4/conv2d_1/bias/Read/ReadVariableOp@module_wrapper_6/batch_normalization_1/gamma/Read/ReadVariableOp?module_wrapper_6/batch_normalization_1/beta/Read/ReadVariableOp4module_wrapper_8/conv2d_2/kernel/Read/ReadVariableOp2module_wrapper_8/conv2d_2/bias/Read/ReadVariableOpAmodule_wrapper_10/batch_normalization_2/gamma/Read/ReadVariableOp@module_wrapper_10/batch_normalization_2/beta/Read/ReadVariableOp5module_wrapper_12/conv2d_3/kernel/Read/ReadVariableOp3module_wrapper_12/conv2d_3/bias/Read/ReadVariableOpAmodule_wrapper_14/batch_normalization_3/gamma/Read/ReadVariableOp@module_wrapper_14/batch_normalization_3/beta/Read/ReadVariableOp2module_wrapper_18/dense/kernel/Read/ReadVariableOp0module_wrapper_18/dense/bias/Read/ReadVariableOpAmodule_wrapper_20/batch_normalization_4/gamma/Read/ReadVariableOp@module_wrapper_20/batch_normalization_4/beta/Read/ReadVariableOp4module_wrapper_22/dense_1/kernel/Read/ReadVariableOp2module_wrapper_22/dense_1/bias/Read/ReadVariableOpDmodule_wrapper_2/batch_normalization/moving_mean/Read/ReadVariableOpHmodule_wrapper_2/batch_normalization/moving_variance/Read/ReadVariableOpFmodule_wrapper_6/batch_normalization_1/moving_mean/Read/ReadVariableOpJmodule_wrapper_6/batch_normalization_1/moving_variance/Read/ReadVariableOpGmodule_wrapper_10/batch_normalization_2/moving_mean/Read/ReadVariableOpKmodule_wrapper_10/batch_normalization_2/moving_variance/Read/ReadVariableOpGmodule_wrapper_14/batch_normalization_3/moving_mean/Read/ReadVariableOpKmodule_wrapper_14/batch_normalization_3/moving_variance/Read/ReadVariableOpGmodule_wrapper_20/batch_normalization_4/moving_mean/Read/ReadVariableOpKmodule_wrapper_20/batch_normalization_4/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp=SGD/module_wrapper/conv2d/kernel/momentum/Read/ReadVariableOp;SGD/module_wrapper/conv2d/bias/momentum/Read/ReadVariableOpKSGD/module_wrapper_2/batch_normalization/gamma/momentum/Read/ReadVariableOpJSGD/module_wrapper_2/batch_normalization/beta/momentum/Read/ReadVariableOpASGD/module_wrapper_4/conv2d_1/kernel/momentum/Read/ReadVariableOp?SGD/module_wrapper_4/conv2d_1/bias/momentum/Read/ReadVariableOpMSGD/module_wrapper_6/batch_normalization_1/gamma/momentum/Read/ReadVariableOpLSGD/module_wrapper_6/batch_normalization_1/beta/momentum/Read/ReadVariableOpASGD/module_wrapper_8/conv2d_2/kernel/momentum/Read/ReadVariableOp?SGD/module_wrapper_8/conv2d_2/bias/momentum/Read/ReadVariableOpNSGD/module_wrapper_10/batch_normalization_2/gamma/momentum/Read/ReadVariableOpMSGD/module_wrapper_10/batch_normalization_2/beta/momentum/Read/ReadVariableOpBSGD/module_wrapper_12/conv2d_3/kernel/momentum/Read/ReadVariableOp@SGD/module_wrapper_12/conv2d_3/bias/momentum/Read/ReadVariableOpNSGD/module_wrapper_14/batch_normalization_3/gamma/momentum/Read/ReadVariableOpMSGD/module_wrapper_14/batch_normalization_3/beta/momentum/Read/ReadVariableOp?SGD/module_wrapper_18/dense/kernel/momentum/Read/ReadVariableOp=SGD/module_wrapper_18/dense/bias/momentum/Read/ReadVariableOpNSGD/module_wrapper_20/batch_normalization_4/gamma/momentum/Read/ReadVariableOpMSGD/module_wrapper_20/batch_normalization_4/beta/momentum/Read/ReadVariableOpASGD/module_wrapper_22/dense_1/kernel/momentum/Read/ReadVariableOp?SGD/module_wrapper_22/dense_1/bias/momentum/Read/ReadVariableOpConst*K
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
GPU 2J 8В *'
f"R 
__inference__traced_save_98742
√
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameSGD/iter	SGD/decaySGD/learning_rateSGD/momentummodule_wrapper/conv2d/kernelmodule_wrapper/conv2d/bias*module_wrapper_2/batch_normalization/gamma)module_wrapper_2/batch_normalization/beta module_wrapper_4/conv2d_1/kernelmodule_wrapper_4/conv2d_1/bias,module_wrapper_6/batch_normalization_1/gamma+module_wrapper_6/batch_normalization_1/beta module_wrapper_8/conv2d_2/kernelmodule_wrapper_8/conv2d_2/bias-module_wrapper_10/batch_normalization_2/gamma,module_wrapper_10/batch_normalization_2/beta!module_wrapper_12/conv2d_3/kernelmodule_wrapper_12/conv2d_3/bias-module_wrapper_14/batch_normalization_3/gamma,module_wrapper_14/batch_normalization_3/betamodule_wrapper_18/dense/kernelmodule_wrapper_18/dense/bias-module_wrapper_20/batch_normalization_4/gamma,module_wrapper_20/batch_normalization_4/beta module_wrapper_22/dense_1/kernelmodule_wrapper_22/dense_1/bias0module_wrapper_2/batch_normalization/moving_mean4module_wrapper_2/batch_normalization/moving_variance2module_wrapper_6/batch_normalization_1/moving_mean6module_wrapper_6/batch_normalization_1/moving_variance3module_wrapper_10/batch_normalization_2/moving_mean7module_wrapper_10/batch_normalization_2/moving_variance3module_wrapper_14/batch_normalization_3/moving_mean7module_wrapper_14/batch_normalization_3/moving_variance3module_wrapper_20/batch_normalization_4/moving_mean7module_wrapper_20/batch_normalization_4/moving_variancetotalcounttotal_1count_1)SGD/module_wrapper/conv2d/kernel/momentum'SGD/module_wrapper/conv2d/bias/momentum7SGD/module_wrapper_2/batch_normalization/gamma/momentum6SGD/module_wrapper_2/batch_normalization/beta/momentum-SGD/module_wrapper_4/conv2d_1/kernel/momentum+SGD/module_wrapper_4/conv2d_1/bias/momentum9SGD/module_wrapper_6/batch_normalization_1/gamma/momentum8SGD/module_wrapper_6/batch_normalization_1/beta/momentum-SGD/module_wrapper_8/conv2d_2/kernel/momentum+SGD/module_wrapper_8/conv2d_2/bias/momentum:SGD/module_wrapper_10/batch_normalization_2/gamma/momentum9SGD/module_wrapper_10/batch_normalization_2/beta/momentum.SGD/module_wrapper_12/conv2d_3/kernel/momentum,SGD/module_wrapper_12/conv2d_3/bias/momentum:SGD/module_wrapper_14/batch_normalization_3/gamma/momentum9SGD/module_wrapper_14/batch_normalization_3/beta/momentum+SGD/module_wrapper_18/dense/kernel/momentum)SGD/module_wrapper_18/dense/bias/momentum:SGD/module_wrapper_20/batch_normalization_4/gamma/momentum9SGD/module_wrapper_20/batch_normalization_4/beta/momentum-SGD/module_wrapper_22/dense_1/kernel/momentum+SGD/module_wrapper_22/dense_1/bias/momentum*J
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_98938°ў
и
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_96373

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
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
П
и
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_94715

args_0A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@
identityИвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╛
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2
conv2d_1/BiasAdd╕
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
ю
g
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_94761

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
л
h
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_97846

args_0
identityn
activation_3/EluEluargs_0*
T0*0
_output_shapes
:         А2
activation_3/Elu{
IdentityIdentityactivation_3/Elu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
ж
g
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_97711

args_0
identitym
activation_2/EluEluargs_0*
T0*/
_output_shapes
:         @2
activation_2/Eluz
IdentityIdentityactivation_2/Elu:activations:0*
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
h
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_98205

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
И
Щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_96263

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
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
Ч
л
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_97821

args_0B
'conv2d_3_conv2d_readvariableop_resource:@А7
(conv2d_3_biasadd_readvariableop_resource:	А
identityИвconv2d_3/BiasAdd/ReadVariableOpвconv2d_3/Conv2D/ReadVariableOp▒
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_3/Conv2D/ReadVariableOp┐
conv2d_3/Conv2DConv2Dargs_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_3/Conv2Dи
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpн
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_3/BiasAdd╣
IdentityIdentityconv2d_3/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

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
╤
е
0__inference_module_wrapper_4_layer_call_fn_97522

args_0!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCallГ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_947152
StatefulPartitionedCallЦ
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
═
г
.__inference_module_wrapper_layer_call_fn_97391

args_0!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallБ
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
GPU 2J 8В *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_956762
StatefulPartitionedCallЦ
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
Э
Ю
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_95083

args_09
&dense_1_matmul_readvariableop_resource:	А5
'dense_1_biasadd_readvariableop_resource:
identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpж
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_1/MatMul/ReadVariableOpЛ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddн
IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
Л
h
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_98040

args_0
identityf
activation_4/EluEluargs_0*
T0*(
_output_shapes
:         А2
activation_4/Elus
IdentityIdentityactivation_4/Elu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
ж
│
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_94937

args_0A
2batch_normalization_4_cast_readvariableop_resource:	АC
4batch_normalization_4_cast_1_readvariableop_resource:	АC
4batch_normalization_4_cast_2_readvariableop_resource:	АC
4batch_normalization_4_cast_3_readvariableop_resource:	А
identityИв)batch_normalization_4/Cast/ReadVariableOpв+batch_normalization_4/Cast_1/ReadVariableOpв+batch_normalization_4/Cast_2/ReadVariableOpв+batch_normalization_4/Cast_3/ReadVariableOp╞
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp╠
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOp╠
+batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+batch_normalization_4/Cast_2/ReadVariableOp╠
+batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+batch_normalization_4/Cast_3/ReadVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/y▐
#batch_normalization_4/batchnorm/addAddV23batch_normalization_4/Cast_1/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/addж
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/Rsqrt╫
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/mul╣
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_4/batchnorm/mul_1╫
%batch_normalization_4/batchnorm/mul_2Mul1batch_normalization_4/Cast/ReadVariableOp:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/mul_2╫
#batch_normalization_4/batchnorm/subSub3batch_normalization_4/Cast_2/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/sub▐
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_4/batchnorm/add_1┤
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp,^batch_normalization_4/Cast_2/ReadVariableOp,^batch_normalization_4/Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2Z
+batch_normalization_4/Cast_2/ReadVariableOp+batch_normalization_4/Cast_2/ReadVariableOp2Z
+batch_normalization_4/Cast_3/ReadVariableOp+batch_normalization_4/Cast_3/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
Щ
╞
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_94746

args_0;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identityИв5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
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
epsilon%oГ:*
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
У
╦
0__inference_module_wrapper_2_layer_call_fn_97457

args_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЫ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_956272
StatefulPartitionedCallЦ
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
С
h
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_94952

args_0
identityo
dropout_1/IdentityIdentityargs_0*
T0*(
_output_shapes
:         А2
dropout_1/Identityp
IdentityIdentitydropout_1/Identity:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
Ч
л
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_95358

args_0B
'conv2d_3_conv2d_readvariableop_resource:@А7
(conv2d_3_biasadd_readvariableop_resource:	А
identityИвconv2d_3/BiasAdd/ReadVariableOpвconv2d_3/Conv2D/ReadVariableOp▒
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_3/Conv2D/ReadVariableOp┐
conv2d_3/Conv2DConv2Dargs_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_3/Conv2Dи
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpн
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_3/BiasAdd╣
IdentityIdentityconv2d_3/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

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
ъ
L
0__inference_module_wrapper_3_layer_call_fn_97498

args_0
identity╤
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_947032
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
╛
k
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_95256

args_0
identityИs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/dropout/ConstФ
dropout/dropout/MulMulargs_0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout/dropout/Muld
dropout/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout/dropout/Shape╒
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2 
dropout/dropout/GreaterEqual/yч
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2
dropout/dropout/GreaterEqualа
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout/dropout/Castг
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout/dropout/Mul_1v
IdentityIdentitydropout/dropout/Mul_1:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
є
h
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_94877

args_0
identity│
max_pooling2d_3/MaxPoolMaxPoolargs_0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool}
IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
У
h
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_98200

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
ъ
Ы
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_97913

args_0<
-batch_normalization_3_readvariableop_resource:	А>
/batch_normalization_3_readvariableop_1_resource:	АM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв$batch_normalization_3/AssignNewValueв&batch_normalization_3/AssignNewValue_1в5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1╖
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization_3/ReadVariableOp╜
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&batch_normalization_3/ReadVariableOp_1ъ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpЁ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1с
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
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
&batch_normalization_3/AssignNewValue_1Щ
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
П
и
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_97691

args_0A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@
identityИвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOp░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp╛
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_2/Conv2Dз
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpм
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAdd╕
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
ж
g
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_95439

args_0
identitym
activation_2/EluEluargs_0*
T0*/
_output_shapes
:         @2
activation_2/Eluz
IdentityIdentityactivation_2/Elu:activations:0*
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
ш
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_97513

args_0
identityо
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
а
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_94668

args_0
identityi
activation/EluEluargs_0*
T0*/
_output_shapes
:         dd 2
activation/Elux
IdentityIdentityactivation/Elu:activations:0*
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
╫
и
1__inference_module_wrapper_12_layer_call_fn_97811

args_0"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_953582
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

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
ъ
L
0__inference_module_wrapper_1_layer_call_fn_97421

args_0
identity╤
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_956512
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
З
╘
5__inference_batch_normalization_4_layer_call_fn_98466

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_968172
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Л
h
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_98035

args_0
identityf
activation_4/EluEluargs_0*
T0*(
_output_shapes
:         А2
activation_4/Elus
IdentityIdentityactivation_4/Elu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
▌
Ю
I__inference_module_wrapper_layer_call_and_return_conditional_losses_95676

args_0?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp╕
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpд
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
╛
┐
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_96583

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1Р
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
Ч
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_95415

args_0;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identityИв$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1╢
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
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
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
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
&batch_normalization_2/AssignNewValue_1Ш
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
Э
Ю
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_98175

args_09
&dense_1_matmul_readvariableop_resource:	А5
'dense_1_biasadd_readvariableop_resource:
identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpж
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_1/MatMul/ReadVariableOpЛ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddн
IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
╝
╜
N__inference_batch_normalization_layer_call_and_return_conditional_losses_96307

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1Р
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
ъ
L
0__inference_module_wrapper_7_layer_call_fn_97638

args_0
identity╤
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_947612
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
а
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_95651

args_0
identityi
activation/EluEluargs_0*
T0*/
_output_shapes
:         dd 2
activation/Elux
IdentityIdentityactivation/Elu:activations:0*
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
╨Л
Х$
__inference__traced_save_98742
file_prefix'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop;
7savev2_module_wrapper_conv2d_kernel_read_readvariableop9
5savev2_module_wrapper_conv2d_bias_read_readvariableopI
Esavev2_module_wrapper_2_batch_normalization_gamma_read_readvariableopH
Dsavev2_module_wrapper_2_batch_normalization_beta_read_readvariableop?
;savev2_module_wrapper_4_conv2d_1_kernel_read_readvariableop=
9savev2_module_wrapper_4_conv2d_1_bias_read_readvariableopK
Gsavev2_module_wrapper_6_batch_normalization_1_gamma_read_readvariableopJ
Fsavev2_module_wrapper_6_batch_normalization_1_beta_read_readvariableop?
;savev2_module_wrapper_8_conv2d_2_kernel_read_readvariableop=
9savev2_module_wrapper_8_conv2d_2_bias_read_readvariableopL
Hsavev2_module_wrapper_10_batch_normalization_2_gamma_read_readvariableopK
Gsavev2_module_wrapper_10_batch_normalization_2_beta_read_readvariableop@
<savev2_module_wrapper_12_conv2d_3_kernel_read_readvariableop>
:savev2_module_wrapper_12_conv2d_3_bias_read_readvariableopL
Hsavev2_module_wrapper_14_batch_normalization_3_gamma_read_readvariableopK
Gsavev2_module_wrapper_14_batch_normalization_3_beta_read_readvariableop=
9savev2_module_wrapper_18_dense_kernel_read_readvariableop;
7savev2_module_wrapper_18_dense_bias_read_readvariableopL
Hsavev2_module_wrapper_20_batch_normalization_4_gamma_read_readvariableopK
Gsavev2_module_wrapper_20_batch_normalization_4_beta_read_readvariableop?
;savev2_module_wrapper_22_dense_1_kernel_read_readvariableop=
9savev2_module_wrapper_22_dense_1_bias_read_readvariableopO
Ksavev2_module_wrapper_2_batch_normalization_moving_mean_read_readvariableopS
Osavev2_module_wrapper_2_batch_normalization_moving_variance_read_readvariableopQ
Msavev2_module_wrapper_6_batch_normalization_1_moving_mean_read_readvariableopU
Qsavev2_module_wrapper_6_batch_normalization_1_moving_variance_read_readvariableopR
Nsavev2_module_wrapper_10_batch_normalization_2_moving_mean_read_readvariableopV
Rsavev2_module_wrapper_10_batch_normalization_2_moving_variance_read_readvariableopR
Nsavev2_module_wrapper_14_batch_normalization_3_moving_mean_read_readvariableopV
Rsavev2_module_wrapper_14_batch_normalization_3_moving_variance_read_readvariableopR
Nsavev2_module_wrapper_20_batch_normalization_4_moving_mean_read_readvariableopV
Rsavev2_module_wrapper_20_batch_normalization_4_moving_variance_read_readvariableop$
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

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*П
valueЕBВ?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЙ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*У
valueЙBЖ?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesк#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop7savev2_module_wrapper_conv2d_kernel_read_readvariableop5savev2_module_wrapper_conv2d_bias_read_readvariableopEsavev2_module_wrapper_2_batch_normalization_gamma_read_readvariableopDsavev2_module_wrapper_2_batch_normalization_beta_read_readvariableop;savev2_module_wrapper_4_conv2d_1_kernel_read_readvariableop9savev2_module_wrapper_4_conv2d_1_bias_read_readvariableopGsavev2_module_wrapper_6_batch_normalization_1_gamma_read_readvariableopFsavev2_module_wrapper_6_batch_normalization_1_beta_read_readvariableop;savev2_module_wrapper_8_conv2d_2_kernel_read_readvariableop9savev2_module_wrapper_8_conv2d_2_bias_read_readvariableopHsavev2_module_wrapper_10_batch_normalization_2_gamma_read_readvariableopGsavev2_module_wrapper_10_batch_normalization_2_beta_read_readvariableop<savev2_module_wrapper_12_conv2d_3_kernel_read_readvariableop:savev2_module_wrapper_12_conv2d_3_bias_read_readvariableopHsavev2_module_wrapper_14_batch_normalization_3_gamma_read_readvariableopGsavev2_module_wrapper_14_batch_normalization_3_beta_read_readvariableop9savev2_module_wrapper_18_dense_kernel_read_readvariableop7savev2_module_wrapper_18_dense_bias_read_readvariableopHsavev2_module_wrapper_20_batch_normalization_4_gamma_read_readvariableopGsavev2_module_wrapper_20_batch_normalization_4_beta_read_readvariableop;savev2_module_wrapper_22_dense_1_kernel_read_readvariableop9savev2_module_wrapper_22_dense_1_bias_read_readvariableopKsavev2_module_wrapper_2_batch_normalization_moving_mean_read_readvariableopOsavev2_module_wrapper_2_batch_normalization_moving_variance_read_readvariableopMsavev2_module_wrapper_6_batch_normalization_1_moving_mean_read_readvariableopQsavev2_module_wrapper_6_batch_normalization_1_moving_variance_read_readvariableopNsavev2_module_wrapper_10_batch_normalization_2_moving_mean_read_readvariableopRsavev2_module_wrapper_10_batch_normalization_2_moving_variance_read_readvariableopNsavev2_module_wrapper_14_batch_normalization_3_moving_mean_read_readvariableopRsavev2_module_wrapper_14_batch_normalization_3_moving_variance_read_readvariableopNsavev2_module_wrapper_20_batch_normalization_4_moving_mean_read_readvariableopRsavev2_module_wrapper_20_batch_normalization_4_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopDsavev2_sgd_module_wrapper_conv2d_kernel_momentum_read_readvariableopBsavev2_sgd_module_wrapper_conv2d_bias_momentum_read_readvariableopRsavev2_sgd_module_wrapper_2_batch_normalization_gamma_momentum_read_readvariableopQsavev2_sgd_module_wrapper_2_batch_normalization_beta_momentum_read_readvariableopHsavev2_sgd_module_wrapper_4_conv2d_1_kernel_momentum_read_readvariableopFsavev2_sgd_module_wrapper_4_conv2d_1_bias_momentum_read_readvariableopTsavev2_sgd_module_wrapper_6_batch_normalization_1_gamma_momentum_read_readvariableopSsavev2_sgd_module_wrapper_6_batch_normalization_1_beta_momentum_read_readvariableopHsavev2_sgd_module_wrapper_8_conv2d_2_kernel_momentum_read_readvariableopFsavev2_sgd_module_wrapper_8_conv2d_2_bias_momentum_read_readvariableopUsavev2_sgd_module_wrapper_10_batch_normalization_2_gamma_momentum_read_readvariableopTsavev2_sgd_module_wrapper_10_batch_normalization_2_beta_momentum_read_readvariableopIsavev2_sgd_module_wrapper_12_conv2d_3_kernel_momentum_read_readvariableopGsavev2_sgd_module_wrapper_12_conv2d_3_bias_momentum_read_readvariableopUsavev2_sgd_module_wrapper_14_batch_normalization_3_gamma_momentum_read_readvariableopTsavev2_sgd_module_wrapper_14_batch_normalization_3_beta_momentum_read_readvariableopFsavev2_sgd_module_wrapper_18_dense_kernel_momentum_read_readvariableopDsavev2_sgd_module_wrapper_18_dense_bias_momentum_read_readvariableopUsavev2_sgd_module_wrapper_20_batch_normalization_4_gamma_momentum_read_readvariableopTsavev2_sgd_module_wrapper_20_batch_normalization_4_beta_momentum_read_readvariableopHsavev2_sgd_module_wrapper_22_dense_1_kernel_momentum_read_readvariableopFsavev2_sgd_module_wrapper_22_dense_1_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
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
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*ї
_input_shapesу
р: : : : : : : : : : @:@:@:@:@@:@:@:@:@А:А:А:А:
АА:А:А:А:	А:: : :@:@:@:@:А:А:А:А: : : : : : : : : @:@:@:@:@@:@:@:@:@А:А:А:А:
АА:А:А:А:	А:: 2(
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
: :,	(
&
_output_shapes
: @: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
:: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:  

_output_shapes
:@:!!

_output_shapes	
:А:!"

_output_shapes	
:А:!#

_output_shapes	
:А:!$

_output_shapes	
:А:%
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
:@А:!6

_output_shapes	
:А:!7

_output_shapes	
:А:!8

_output_shapes	
:А:&9"
 
_output_shapes
:
АА:!:

_output_shapes	
:А:!;

_output_shapes	
:А:!<

_output_shapes	
:А:%=!

_output_shapes
:	А: >

_output_shapes
::?

_output_shapes
: 
ж
g
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_95545

args_0
identitym
activation_1/EluEluargs_0*
T0*/
_output_shapes
:         !!@2
activation_1/Eluz
IdentityIdentityactivation_1/Elu:activations:0*
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
Ё
M
1__inference_module_wrapper_13_layer_call_fn_97836

args_0
identity╙
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_948422
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
▌
Ю
I__inference_module_wrapper_layer_call_and_return_conditional_losses_97411

args_0?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp╕
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpд
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
Ч
л
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_97831

args_0B
'conv2d_3_conv2d_readvariableop_resource:@А7
(conv2d_3_biasadd_readvariableop_resource:	А
identityИвconv2d_3/BiasAdd/ReadVariableOpвconv2d_3/Conv2D/ReadVariableOp▒
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_3/Conv2D/ReadVariableOp┐
conv2d_3/Conv2DConv2Dargs_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_3/Conv2Dи
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpн
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_3/BiasAdd╣
IdentityIdentityconv2d_3/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

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
╨
M
1__inference_module_wrapper_21_layer_call_fn_98125

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
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_949522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
П
и
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_97551

args_0A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@
identityИвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╛
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2
conv2d_1/BiasAdd╕
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
ФЗ
ч)
 __inference__wrapped_model_94640
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
Dsequential_module_wrapper_12_conv2d_3_conv2d_readvariableop_resource:@АT
Esequential_module_wrapper_12_conv2d_3_biasadd_readvariableop_resource:	АY
Jsequential_module_wrapper_14_batch_normalization_3_readvariableop_resource:	А[
Lsequential_module_wrapper_14_batch_normalization_3_readvariableop_1_resource:	Аj
[sequential_module_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	Аl
]sequential_module_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АU
Asequential_module_wrapper_18_dense_matmul_readvariableop_resource:
ААQ
Bsequential_module_wrapper_18_dense_biasadd_readvariableop_resource:	А^
Osequential_module_wrapper_20_batch_normalization_4_cast_readvariableop_resource:	А`
Qsequential_module_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource:	А`
Qsequential_module_wrapper_20_batch_normalization_4_cast_2_readvariableop_resource:	А`
Qsequential_module_wrapper_20_batch_normalization_4_cast_3_readvariableop_resource:	АV
Csequential_module_wrapper_22_dense_1_matmul_readvariableop_resource:	АR
Dsequential_module_wrapper_22_dense_1_biasadd_readvariableop_resource:
identityИв7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOpв6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOpвRsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвTsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1вAsequential/module_wrapper_10/batch_normalization_2/ReadVariableOpвCsequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1в<sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpв;sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpвRsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвTsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1вAsequential/module_wrapper_14/batch_normalization_3/ReadVariableOpвCsequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1в9sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOpв8sequential/module_wrapper_18/dense/MatMul/ReadVariableOpвOsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpвQsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1в>sequential/module_wrapper_2/batch_normalization/ReadVariableOpв@sequential/module_wrapper_2/batch_normalization/ReadVariableOp_1вFsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpвHsequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpвHsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpвHsequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpв;sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOpв:sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOpв;sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpв:sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpвQsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpвSsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в@sequential/module_wrapper_6/batch_normalization_1/ReadVariableOpвBsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1в;sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpв:sequential/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp°
6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOpReadVariableOp?sequential_module_wrapper_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype028
6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOpФ
'sequential/module_wrapper/conv2d/Conv2DConv2Dmodule_wrapper_input>sequential/module_wrapper/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2)
'sequential/module_wrapper/conv2d/Conv2Dя
7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOpReadVariableOp@sequential_module_wrapper_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOpМ
(sequential/module_wrapper/conv2d/BiasAddBiasAdd0sequential/module_wrapper/conv2d/Conv2D:output:0?sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd 2*
(sequential/module_wrapper/conv2d/BiasAdd╠
*sequential/module_wrapper_1/activation/EluElu1sequential/module_wrapper/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         dd 2,
*sequential/module_wrapper_1/activation/EluД
>sequential/module_wrapper_2/batch_normalization/ReadVariableOpReadVariableOpGsequential_module_wrapper_2_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02@
>sequential/module_wrapper_2/batch_normalization/ReadVariableOpК
@sequential/module_wrapper_2/batch_normalization/ReadVariableOp_1ReadVariableOpIsequential_module_wrapper_2_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@sequential/module_wrapper_2/batch_normalization/ReadVariableOp_1╖
Osequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpXsequential_module_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Q
Osequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp╜
Qsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZsequential_module_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02S
Qsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ь
@sequential/module_wrapper_2/batch_normalization/FusedBatchNormV3FusedBatchNormV38sequential/module_wrapper_1/activation/Elu:activations:0Fsequential/module_wrapper_2/batch_normalization/ReadVariableOp:value:0Hsequential/module_wrapper_2/batch_normalization/ReadVariableOp_1:value:0Wsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ysequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oГ:*
is_training( 2B
@sequential/module_wrapper_2/batch_normalization/FusedBatchNormV3д
1sequential/module_wrapper_3/max_pooling2d/MaxPoolMaxPoolDsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
23
1sequential/module_wrapper_3/max_pooling2d/MaxPoolД
:sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpReadVariableOpCsequential_module_wrapper_4_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02<
:sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp╞
+sequential/module_wrapper_4/conv2d_1/Conv2DConv2D:sequential/module_wrapper_3/max_pooling2d/MaxPool:output:0Bsequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2-
+sequential/module_wrapper_4/conv2d_1/Conv2D√
;sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpDsequential_module_wrapper_4_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpЬ
,sequential/module_wrapper_4/conv2d_1/BiasAddBiasAdd4sequential/module_wrapper_4/conv2d_1/Conv2D:output:0Csequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2.
,sequential/module_wrapper_4/conv2d_1/BiasAdd╘
,sequential/module_wrapper_5/activation_1/EluElu5sequential/module_wrapper_4/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         !!@2.
,sequential/module_wrapper_5/activation_1/EluК
@sequential/module_wrapper_6/batch_normalization_1/ReadVariableOpReadVariableOpIsequential_module_wrapper_6_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@sequential/module_wrapper_6/batch_normalization_1/ReadVariableOpР
Bsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1ReadVariableOpKsequential_module_wrapper_6_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1╜
Qsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpZsequential_module_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02S
Qsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp├
Ssequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\sequential_module_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02U
Ssequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1к
Bsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3:sequential/module_wrapper_5/activation_1/Elu:activations:0Hsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp:value:0Jsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1:value:0Ysequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0[sequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !!@:@:@:@:@:*
epsilon%oГ:*
is_training( 2D
Bsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3к
3sequential/module_wrapper_7/max_pooling2d_1/MaxPoolMaxPoolFsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
25
3sequential/module_wrapper_7/max_pooling2d_1/MaxPoolД
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
+sequential/module_wrapper_8/conv2d_2/Conv2D√
;sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpDsequential_module_wrapper_8_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpЬ
,sequential/module_wrapper_8/conv2d_2/BiasAddBiasAdd4sequential/module_wrapper_8/conv2d_2/Conv2D:output:0Csequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2.
,sequential/module_wrapper_8/conv2d_2/BiasAdd╘
,sequential/module_wrapper_9/activation_2/EluElu5sequential/module_wrapper_8/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2.
,sequential/module_wrapper_9/activation_2/EluН
Asequential/module_wrapper_10/batch_normalization_2/ReadVariableOpReadVariableOpJsequential_module_wrapper_10_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02C
Asequential/module_wrapper_10/batch_normalization_2/ReadVariableOpУ
Csequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1ReadVariableOpLsequential_module_wrapper_10_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Csequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1└
Rsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp[sequential_module_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp╞
Tsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]sequential_module_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02V
Tsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1░
Csequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3:sequential/module_wrapper_9/activation_2/Elu:activations:0Isequential/module_wrapper_10/batch_normalization_2/ReadVariableOp:value:0Ksequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1:value:0Zsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0\sequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2E
Csequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3н
4sequential/module_wrapper_11/max_pooling2d_2/MaxPoolMaxPoolGsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
26
4sequential/module_wrapper_11/max_pooling2d_2/MaxPoolИ
;sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpReadVariableOpDsequential_module_wrapper_12_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02=
;sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp═
,sequential/module_wrapper_12/conv2d_3/Conv2DConv2D=sequential/module_wrapper_11/max_pooling2d_2/MaxPool:output:0Csequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2.
,sequential/module_wrapper_12/conv2d_3/Conv2D 
<sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpEsequential_module_wrapper_12_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02>
<sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpб
-sequential/module_wrapper_12/conv2d_3/BiasAddBiasAdd5sequential/module_wrapper_12/conv2d_3/Conv2D:output:0Dsequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2/
-sequential/module_wrapper_12/conv2d_3/BiasAdd╪
-sequential/module_wrapper_13/activation_3/EluElu6sequential/module_wrapper_12/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:         А2/
-sequential/module_wrapper_13/activation_3/EluО
Asequential/module_wrapper_14/batch_normalization_3/ReadVariableOpReadVariableOpJsequential_module_wrapper_14_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype02C
Asequential/module_wrapper_14/batch_normalization_3/ReadVariableOpФ
Csequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1ReadVariableOpLsequential_module_wrapper_14_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02E
Csequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1┴
Rsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp[sequential_module_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02T
Rsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp╟
Tsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]sequential_module_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02V
Tsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1╢
Csequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3;sequential/module_wrapper_13/activation_3/Elu:activations:0Isequential/module_wrapper_14/batch_normalization_3/ReadVariableOp:value:0Ksequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1:value:0Zsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0\sequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2E
Csequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3о
4sequential/module_wrapper_15/max_pooling2d_3/MaxPoolMaxPoolGsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
26
4sequential/module_wrapper_15/max_pooling2d_3/MaxPoolф
-sequential/module_wrapper_16/dropout/IdentityIdentity=sequential/module_wrapper_15/max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         А2/
-sequential/module_wrapper_16/dropout/Identityй
*sequential/module_wrapper_17/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*sequential/module_wrapper_17/flatten/ConstЗ
,sequential/module_wrapper_17/flatten/ReshapeReshape6sequential/module_wrapper_16/dropout/Identity:output:03sequential/module_wrapper_17/flatten/Const:output:0*
T0*(
_output_shapes
:         А2.
,sequential/module_wrapper_17/flatten/Reshape°
8sequential/module_wrapper_18/dense/MatMul/ReadVariableOpReadVariableOpAsequential_module_wrapper_18_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02:
8sequential/module_wrapper_18/dense/MatMul/ReadVariableOpМ
)sequential/module_wrapper_18/dense/MatMulMatMul5sequential/module_wrapper_17/flatten/Reshape:output:0@sequential/module_wrapper_18/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2+
)sequential/module_wrapper_18/dense/MatMulЎ
9sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOpReadVariableOpBsequential_module_wrapper_18_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOpО
*sequential/module_wrapper_18/dense/BiasAddBiasAdd3sequential/module_wrapper_18/dense/MatMul:product:0Asequential/module_wrapper_18/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*sequential/module_wrapper_18/dense/BiasAdd═
-sequential/module_wrapper_19/activation_4/EluElu3sequential/module_wrapper_18/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2/
-sequential/module_wrapper_19/activation_4/EluЭ
Fsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpReadVariableOpOsequential_module_wrapper_20_batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:А*
dtype02H
Fsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpг
Hsequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpReadVariableOpQsequential_module_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02J
Hsequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpг
Hsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpReadVariableOpQsequential_module_wrapper_20_batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:А*
dtype02J
Hsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpг
Hsequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpReadVariableOpQsequential_module_wrapper_20_batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:А*
dtype02J
Hsequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp═
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2D
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/add/y╥
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/addAddV2Psequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp:value:0Ksequential/module_wrapper_20/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2B
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/add¤
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/RsqrtRsqrtDsequential/module_wrapper_20/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А2D
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/Rsqrt╦
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/mulMulFsequential/module_wrapper_20/batch_normalization_4/batchnorm/Rsqrt:y:0Psequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2B
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/mul┼
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_1Mul;sequential/module_wrapper_19/activation_4/Elu:activations:0Dsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2D
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_1╦
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_2MulNsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp:value:0Dsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2D
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_2╦
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/subSubPsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOp:value:0Fsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2B
@sequential/module_wrapper_20/batch_normalization_4/batchnorm/sub╥
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/add_1AddV2Fsequential/module_wrapper_20/batch_normalization_4/batchnorm/mul_1:z:0Dsequential/module_wrapper_20/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2D
Bsequential/module_wrapper_20/batch_normalization_4/batchnorm/add_1щ
/sequential/module_wrapper_21/dropout_1/IdentityIdentityFsequential/module_wrapper_20/batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         А21
/sequential/module_wrapper_21/dropout_1/Identity¤
:sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOpReadVariableOpCsequential_module_wrapper_22_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02<
:sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOpФ
+sequential/module_wrapper_22/dense_1/MatMulMatMul8sequential/module_wrapper_21/dropout_1/Identity:output:0Bsequential/module_wrapper_22/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2-
+sequential/module_wrapper_22/dense_1/MatMul√
;sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOpReadVariableOpDsequential_module_wrapper_22_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOpХ
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
6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOp6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOp2и
Rsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpRsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2м
Tsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Tsequential/module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12Ж
Asequential/module_wrapper_10/batch_normalization_2/ReadVariableOpAsequential/module_wrapper_10/batch_normalization_2/ReadVariableOp2К
Csequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_1Csequential/module_wrapper_10/batch_normalization_2/ReadVariableOp_12|
<sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp<sequential/module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp2z
;sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp;sequential/module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp2и
Rsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpRsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2м
Tsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Tsequential/module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12Ж
Asequential/module_wrapper_14/batch_normalization_3/ReadVariableOpAsequential/module_wrapper_14/batch_normalization_3/ReadVariableOp2К
Csequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_1Csequential/module_wrapper_14/batch_normalization_3/ReadVariableOp_12v
9sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOp9sequential/module_wrapper_18/dense/BiasAdd/ReadVariableOp2t
8sequential/module_wrapper_18/dense/MatMul/ReadVariableOp8sequential/module_wrapper_18/dense/MatMul/ReadVariableOp2в
Osequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpOsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp2ж
Qsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Qsequential/module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_12А
>sequential/module_wrapper_2/batch_normalization/ReadVariableOp>sequential/module_wrapper_2/batch_normalization/ReadVariableOp2Д
@sequential/module_wrapper_2/batch_normalization/ReadVariableOp_1@sequential/module_wrapper_2/batch_normalization/ReadVariableOp_12Р
Fsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpFsequential/module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp2Ф
Hsequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpHsequential/module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp2Ф
Hsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpHsequential/module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOp2Ф
Hsequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpHsequential/module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp2z
;sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOp;sequential/module_wrapper_22/dense_1/BiasAdd/ReadVariableOp2x
:sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOp:sequential/module_wrapper_22/dense_1/MatMul/ReadVariableOp2z
;sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp;sequential/module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp2x
:sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp:sequential/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp2ж
Qsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpQsequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2к
Ssequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ssequential/module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12Д
@sequential/module_wrapper_6/batch_normalization_1/ReadVariableOp@sequential/module_wrapper_6/batch_normalization_1/ReadVariableOp2И
Bsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_1Bsequential/module_wrapper_6/batch_normalization_1/ReadVariableOp_12z
;sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp;sequential/module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp2x
:sequential/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:sequential/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:e a
/
_output_shapes
:         dd
.
_user_specified_namemodule_wrapper_input
ъ
L
0__inference_module_wrapper_9_layer_call_fn_97701

args_0
identity╤
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_954392
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
¤
╨
1__inference_module_wrapper_20_layer_call_fn_98066

args_0
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_951632
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
а
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_97426

args_0
identityi
activation/EluEluargs_0*
T0*/
_output_shapes
:         dd 2
activation/Elux
IdentityIdentityactivation/Elu:activations:0*
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
я
h
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_94819

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
ъ
L
0__inference_module_wrapper_1_layer_call_fn_97416

args_0
identity╤
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_946682
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
╤
е
0__inference_module_wrapper_4_layer_call_fn_97531

args_0!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCallГ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_955702
StatefulPartitionedCallЦ
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
╛
┐
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_98391

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1Р
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
Э
╨
1__inference_module_wrapper_14_layer_call_fn_97877

args_0
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_953092
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
▌
Ю
I__inference_module_wrapper_layer_call_and_return_conditional_losses_94657

args_0?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp╕
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpд
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
о
k
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_95110

args_0
identityИw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/ConstТ
dropout_1/dropout/MulMulargs_0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_1/dropout/Mulh
dropout_1/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape╙
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yч
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2 
dropout_1/dropout/GreaterEqualЮ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_1/dropout/Castг
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_1/dropout/Mul_1p
IdentityIdentitydropout_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
ъ
L
0__inference_module_wrapper_9_layer_call_fn_97696

args_0
identity╤
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_947842
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
Х
╦
0__inference_module_wrapper_6_layer_call_fn_97584

args_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЭ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_947462
StatefulPartitionedCallЦ
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
ч
В
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_95627

args_09
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: 
identityИв"batch_normalization/AssignNewValueв$batch_normalization/AssignNewValue_1в3batch_normalization/FusedBatchNormV3/ReadVariableOpв5batch_normalization/FusedBatchNormV3/ReadVariableOp_1в"batch_normalization/ReadVariableOpв$batch_normalization/ReadVariableOp_1░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1╨
$batch_normalization/FusedBatchNormV3FusedBatchNormV3args_0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2&
$batch_normalization/FusedBatchNormV3ж
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue▓
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1К
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
У
╦
0__inference_module_wrapper_6_layer_call_fn_97597

args_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЫ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_955212
StatefulPartitionedCallЦ
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
Ъ
Я
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_98435

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
у
╬
3__inference_batch_normalization_layer_call_fn_98218

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCall▓
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
GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_962632
StatefulPartitionedCallи
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
ш
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_95590

args_0
identityо
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
Ё
M
1__inference_module_wrapper_15_layer_call_fn_97918

args_0
identity╙
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_948772
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
▓
б
1__inference_module_wrapper_18_layer_call_fn_98000

args_0
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_952122
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
╝
╜
N__inference_batch_normalization_layer_call_and_return_conditional_losses_98267

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1Р
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
я
h
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_97793

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
х
╨
5__inference_batch_normalization_2_layer_call_fn_98355

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCall▓
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
GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_965832
StatefulPartitionedCallи
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
є
h
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_95272

args_0
identity│
max_pooling2d_3/MaxPoolMaxPoolargs_0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool}
IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
б
h
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_95233

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/ConstА
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         А2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
к
╦
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_97895

args_0<
-batch_normalization_3_readvariableop_resource:	А>
/batch_normalization_3_readvariableop_1_resource:	АM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1╖
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization_3/ReadVariableOp╜
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&batch_normalization_3/ReadVariableOp_1ъ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpЁ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1╙
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3╔
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
о
Я
1__inference_module_wrapper_22_layer_call_fn_98165

args_0
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall№
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_950832
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
жs
у
E__inference_sequential_layer_call_and_return_conditional_losses_96074
module_wrapper_input.
module_wrapper_95985: "
module_wrapper_95987: $
module_wrapper_2_95991: $
module_wrapper_2_95993: $
module_wrapper_2_95995: $
module_wrapper_2_95997: 0
module_wrapper_4_96001: @$
module_wrapper_4_96003:@$
module_wrapper_6_96007:@$
module_wrapper_6_96009:@$
module_wrapper_6_96011:@$
module_wrapper_6_96013:@0
module_wrapper_8_96017:@@$
module_wrapper_8_96019:@%
module_wrapper_10_96023:@%
module_wrapper_10_96025:@%
module_wrapper_10_96027:@%
module_wrapper_10_96029:@2
module_wrapper_12_96033:@А&
module_wrapper_12_96035:	А&
module_wrapper_14_96039:	А&
module_wrapper_14_96041:	А&
module_wrapper_14_96043:	А&
module_wrapper_14_96045:	А+
module_wrapper_18_96051:
АА&
module_wrapper_18_96053:	А&
module_wrapper_20_96057:	А&
module_wrapper_20_96059:	А&
module_wrapper_20_96061:	А&
module_wrapper_20_96063:	А*
module_wrapper_22_96067:	А%
module_wrapper_22_96069:
identityИв&module_wrapper/StatefulPartitionedCallв)module_wrapper_10/StatefulPartitionedCallв)module_wrapper_12/StatefulPartitionedCallв)module_wrapper_14/StatefulPartitionedCallв)module_wrapper_18/StatefulPartitionedCallв(module_wrapper_2/StatefulPartitionedCallв)module_wrapper_20/StatefulPartitionedCallв)module_wrapper_22/StatefulPartitionedCallв(module_wrapper_4/StatefulPartitionedCallв(module_wrapper_6/StatefulPartitionedCallв(module_wrapper_8/StatefulPartitionedCall┼
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_95985module_wrapper_95987*
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
GPU 2J 8В *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_946572(
&module_wrapper/StatefulPartitionedCallЬ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_946682"
 module_wrapper_1/PartitionedCallШ
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0module_wrapper_2_95991module_wrapper_2_95993module_wrapper_2_95995module_wrapper_2_95997*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_946882*
(module_wrapper_2/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_947032"
 module_wrapper_3/PartitionedCallф
(module_wrapper_4/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0module_wrapper_4_96001module_wrapper_4_96003*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_947152*
(module_wrapper_4/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_947262"
 module_wrapper_5/PartitionedCallШ
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_5/PartitionedCall:output:0module_wrapper_6_96007module_wrapper_6_96009module_wrapper_6_96011module_wrapper_6_96013*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_947462*
(module_wrapper_6/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_947612"
 module_wrapper_7/PartitionedCallф
(module_wrapper_8/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_7/PartitionedCall:output:0module_wrapper_8_96017module_wrapper_8_96019*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_947732*
(module_wrapper_8/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_947842"
 module_wrapper_9/PartitionedCallЯ
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_96023module_wrapper_10_96025module_wrapper_10_96027module_wrapper_10_96029*
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_948042+
)module_wrapper_10/StatefulPartitionedCallв
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_948192#
!module_wrapper_11/PartitionedCallы
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_11/PartitionedCall:output:0module_wrapper_12_96033module_wrapper_12_96035*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_948312+
)module_wrapper_12/StatefulPartitionedCallг
!module_wrapper_13/PartitionedCallPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_948422#
!module_wrapper_13/PartitionedCallб
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_13/PartitionedCall:output:0module_wrapper_14_96039module_wrapper_14_96041module_wrapper_14_96043module_wrapper_14_96045*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_948622+
)module_wrapper_14/StatefulPartitionedCallг
!module_wrapper_15/PartitionedCallPartitionedCall2module_wrapper_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_948772#
!module_wrapper_15/PartitionedCallЫ
!module_wrapper_16/PartitionedCallPartitionedCall*module_wrapper_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_948842#
!module_wrapper_16/PartitionedCallУ
!module_wrapper_17/PartitionedCallPartitionedCall*module_wrapper_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_948922#
!module_wrapper_17/PartitionedCallу
)module_wrapper_18/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_17/PartitionedCall:output:0module_wrapper_18_96051module_wrapper_18_96053*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_949042+
)module_wrapper_18/StatefulPartitionedCallЫ
!module_wrapper_19/PartitionedCallPartitionedCall2module_wrapper_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_949152#
!module_wrapper_19/PartitionedCallЩ
)module_wrapper_20/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_19/PartitionedCall:output:0module_wrapper_20_96057module_wrapper_20_96059module_wrapper_20_96061module_wrapper_20_96063*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_949372+
)module_wrapper_20/StatefulPartitionedCallЫ
!module_wrapper_21/PartitionedCallPartitionedCall2module_wrapper_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_949522#
!module_wrapper_21/PartitionedCallт
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_21/PartitionedCall:output:0module_wrapper_22_96067module_wrapper_22_96069*
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_949642+
)module_wrapper_22/StatefulPartitionedCallЪ
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_949752#
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
ч
╨
5__inference_batch_normalization_2_layer_call_fn_98342

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCall┤
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
GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_965392
StatefulPartitionedCallи
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
ъ
L
0__inference_module_wrapper_5_layer_call_fn_97561

args_0
identity╤
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_955452
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
Ё
M
1__inference_module_wrapper_13_layer_call_fn_97841

args_0
identity╙
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_953332
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
Ї

Ш
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_98010

args_08
$dense_matmul_readvariableop_resource:
АА4
%dense_biasadd_readvariableop_resource:	А
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense/MatMul/ReadVariableOpЖ
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/BiasAddи
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
╦
╢
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_97475

args_09
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: 
identityИв3batch_normalization/FusedBatchNormV3/ReadVariableOpв5batch_normalization/FusedBatchNormV3/ReadVariableOp_1в"batch_normalization/ReadVariableOpв$batch_normalization/ReadVariableOp_1░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
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
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3╛
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
Щ
╞
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_97615

args_0;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identityИв5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
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
epsilon%oГ:*
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
┐
▄
*__inference_sequential_layer_call_fn_97093

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

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23:
АА

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:
identityИвStatefulPartitionedCallЖ
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
GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_958462
StatefulPartitionedCallО
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
П
и
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_95464

args_0A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@
identityИвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOp░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp╛
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_2/Conv2Dз
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpм
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAdd╕
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
э
╘
5__inference_batch_normalization_3_layer_call_fn_98417

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_967212
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ъ
L
0__inference_module_wrapper_5_layer_call_fn_97556

args_0
identity╤
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_947262
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
ъ
L
0__inference_module_wrapper_3_layer_call_fn_97503

args_0
identity╤
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_955902
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
▌
Ю
I__inference_module_wrapper_layer_call_and_return_conditional_losses_97401

args_0?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp╕
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd *
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpд
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
╛
k
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_97960

args_0
identityИs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/dropout/ConstФ
dropout/dropout/MulMulargs_0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout/dropout/Muld
dropout/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout/dropout/Shape╒
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2 
dropout/dropout/GreaterEqual/yч
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2
dropout/dropout/GreaterEqualа
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout/dropout/Castг
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout/dropout/Mul_1v
IdentityIdentitydropout/dropout/Mul_1:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
б
h
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_97982

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/ConstА
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         А2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
к
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_96787

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
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
К
Ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_98311

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
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
╒
K
/__inference_max_pooling2d_3_layer_call_fn_96793

inputs
identityы
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
GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_967872
PartitionedCallП
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
ж
g
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_94784

args_0
identitym
activation_2/EluEluargs_0*
T0*/
_output_shapes
:         @2
activation_2/Eluz
IdentityIdentityactivation_2/Elu:activations:0*
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
в)
╒
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_96877

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А+
cast_readvariableop_resource:	А-
cast_1_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвCast/ReadVariableOpвCast_1/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Д
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1А
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
П
и
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_95570

args_0A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@
identityИвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╛
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2
conv2d_1/BiasAdd╕
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
ц
З
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_96817

inputs+
cast_readvariableop_resource:	А-
cast_1_readvariableop_resource:	А-
cast_2_readvariableop_resource:	А-
cast_3_readvariableop_resource:	А
identityИвCast/ReadVariableOpвCast_1/ReadVariableOpвCast_2/ReadVariableOpвCast_3/ReadVariableOpД
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast_1/ReadVariableOpК
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast_2/ReadVariableOpК
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЖ
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1╞
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
У
h
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_94975

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
ч
В
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_97493

args_09
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: 
identityИв"batch_normalization/AssignNewValueв$batch_normalization/AssignNewValue_1в3batch_normalization/FusedBatchNormV3/ReadVariableOpв5batch_normalization/FusedBatchNormV3/ReadVariableOp_1в"batch_normalization/ReadVariableOpв$batch_normalization/ReadVariableOp_1░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1╨
$batch_normalization/FusedBatchNormV3FusedBatchNormV3args_0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2&
$batch_normalization/FusedBatchNormV3ж
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue▓
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1К
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
ж
g
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_97571

args_0
identitym
activation_1/EluEluargs_0*
T0*/
_output_shapes
:         !!@2
activation_1/Eluz
IdentityIdentityactivation_1/Elu:activations:0*
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
Ц
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_97633

args_0;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identityИв$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
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
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
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
&batch_normalization_1/AssignNewValue_1Ш
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
щ
ъ
*__inference_sequential_layer_call_fn_95982
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

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23:
АА

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:
identityИвStatefulPartitionedCallФ
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
GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_958462
StatefulPartitionedCallО
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
к
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_96511

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
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
╟
у
#__inference_signature_wrapper_96241
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

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23:
АА

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:
identityИвStatefulPartitionedCall∙
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
GPU 2J 8В *)
f$R"
 __inference__wrapped_model_946402
StatefulPartitionedCallО
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
┌
Ч
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_97773

args_0;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identityИв$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1╢
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
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
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
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
&batch_normalization_2/AssignNewValue_1Ш
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
Ъ
Я
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_96677

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╛
┐
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_98329

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1Р
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
▄
j
1__inference_module_wrapper_21_layer_call_fn_98130

args_0
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_951102
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
·Щ
╨0
!__inference__traced_restore_98938
file_prefix#
assignvariableop_sgd_iter:	 &
assignvariableop_1_sgd_decay: .
$assignvariableop_2_sgd_learning_rate: )
assignvariableop_3_sgd_momentum: I
/assignvariableop_4_module_wrapper_conv2d_kernel: ;
-assignvariableop_5_module_wrapper_conv2d_bias: K
=assignvariableop_6_module_wrapper_2_batch_normalization_gamma: J
<assignvariableop_7_module_wrapper_2_batch_normalization_beta: M
3assignvariableop_8_module_wrapper_4_conv2d_1_kernel: @?
1assignvariableop_9_module_wrapper_4_conv2d_1_bias:@N
@assignvariableop_10_module_wrapper_6_batch_normalization_1_gamma:@M
?assignvariableop_11_module_wrapper_6_batch_normalization_1_beta:@N
4assignvariableop_12_module_wrapper_8_conv2d_2_kernel:@@@
2assignvariableop_13_module_wrapper_8_conv2d_2_bias:@O
Aassignvariableop_14_module_wrapper_10_batch_normalization_2_gamma:@N
@assignvariableop_15_module_wrapper_10_batch_normalization_2_beta:@P
5assignvariableop_16_module_wrapper_12_conv2d_3_kernel:@АB
3assignvariableop_17_module_wrapper_12_conv2d_3_bias:	АP
Aassignvariableop_18_module_wrapper_14_batch_normalization_3_gamma:	АO
@assignvariableop_19_module_wrapper_14_batch_normalization_3_beta:	АF
2assignvariableop_20_module_wrapper_18_dense_kernel:
АА?
0assignvariableop_21_module_wrapper_18_dense_bias:	АP
Aassignvariableop_22_module_wrapper_20_batch_normalization_4_gamma:	АO
@assignvariableop_23_module_wrapper_20_batch_normalization_4_beta:	АG
4assignvariableop_24_module_wrapper_22_dense_1_kernel:	А@
2assignvariableop_25_module_wrapper_22_dense_1_bias:R
Dassignvariableop_26_module_wrapper_2_batch_normalization_moving_mean: V
Hassignvariableop_27_module_wrapper_2_batch_normalization_moving_variance: T
Fassignvariableop_28_module_wrapper_6_batch_normalization_1_moving_mean:@X
Jassignvariableop_29_module_wrapper_6_batch_normalization_1_moving_variance:@U
Gassignvariableop_30_module_wrapper_10_batch_normalization_2_moving_mean:@Y
Kassignvariableop_31_module_wrapper_10_batch_normalization_2_moving_variance:@V
Gassignvariableop_32_module_wrapper_14_batch_normalization_3_moving_mean:	АZ
Kassignvariableop_33_module_wrapper_14_batch_normalization_3_moving_variance:	АV
Gassignvariableop_34_module_wrapper_20_batch_normalization_4_moving_mean:	АZ
Kassignvariableop_35_module_wrapper_20_batch_normalization_4_moving_variance:	А#
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
Bassignvariableop_52_sgd_module_wrapper_12_conv2d_3_kernel_momentum:@АO
@assignvariableop_53_sgd_module_wrapper_12_conv2d_3_bias_momentum:	А]
Nassignvariableop_54_sgd_module_wrapper_14_batch_normalization_3_gamma_momentum:	А\
Massignvariableop_55_sgd_module_wrapper_14_batch_normalization_3_beta_momentum:	АS
?assignvariableop_56_sgd_module_wrapper_18_dense_kernel_momentum:
ААL
=assignvariableop_57_sgd_module_wrapper_18_dense_bias_momentum:	А]
Nassignvariableop_58_sgd_module_wrapper_20_batch_normalization_4_gamma_momentum:	А\
Massignvariableop_59_sgd_module_wrapper_20_batch_normalization_4_beta_momentum:	АT
Aassignvariableop_60_sgd_module_wrapper_22_dense_1_kernel_momentum:	АM
?assignvariableop_61_sgd_module_wrapper_22_dense_1_bias_momentum:
identity_63ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Г
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*П
valueЕBВ?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesП
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*У
valueЙBЖ?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesщ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Т
_output_shapes 
№:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

IdentityШ
AssignVariableOpAssignVariableOpassignvariableop_sgd_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1б
AssignVariableOp_1AssignVariableOpassignvariableop_1_sgd_decayIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2й
AssignVariableOp_2AssignVariableOp$assignvariableop_2_sgd_learning_rateIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3д
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

Identity_8╕
AssignVariableOp_8AssignVariableOp3assignvariableop_8_module_wrapper_4_conv2d_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╢
AssignVariableOp_9AssignVariableOp1assignvariableop_9_module_wrapper_4_conv2d_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╚
AssignVariableOp_10AssignVariableOp@assignvariableop_10_module_wrapper_6_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╟
AssignVariableOp_11AssignVariableOp?assignvariableop_11_module_wrapper_6_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╝
AssignVariableOp_12AssignVariableOp4assignvariableop_12_module_wrapper_8_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13║
AssignVariableOp_13AssignVariableOp2assignvariableop_13_module_wrapper_8_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14╔
AssignVariableOp_14AssignVariableOpAassignvariableop_14_module_wrapper_10_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╚
AssignVariableOp_15AssignVariableOp@assignvariableop_15_module_wrapper_10_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╜
AssignVariableOp_16AssignVariableOp5assignvariableop_16_module_wrapper_12_conv2d_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╗
AssignVariableOp_17AssignVariableOp3assignvariableop_17_module_wrapper_12_conv2d_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╔
AssignVariableOp_18AssignVariableOpAassignvariableop_18_module_wrapper_14_batch_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╚
AssignVariableOp_19AssignVariableOp@assignvariableop_19_module_wrapper_14_batch_normalization_3_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20║
AssignVariableOp_20AssignVariableOp2assignvariableop_20_module_wrapper_18_dense_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╕
AssignVariableOp_21AssignVariableOp0assignvariableop_21_module_wrapper_18_dense_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╔
AssignVariableOp_22AssignVariableOpAassignvariableop_22_module_wrapper_20_batch_normalization_4_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╚
AssignVariableOp_23AssignVariableOp@assignvariableop_23_module_wrapper_20_batch_normalization_4_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╝
AssignVariableOp_24AssignVariableOp4assignvariableop_24_module_wrapper_22_dense_1_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25║
AssignVariableOp_25AssignVariableOp2assignvariableop_25_module_wrapper_22_dense_1_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╠
AssignVariableOp_26AssignVariableOpDassignvariableop_26_module_wrapper_2_batch_normalization_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╨
AssignVariableOp_27AssignVariableOpHassignvariableop_27_module_wrapper_2_batch_normalization_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╬
AssignVariableOp_28AssignVariableOpFassignvariableop_28_module_wrapper_6_batch_normalization_1_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╥
AssignVariableOp_29AssignVariableOpJassignvariableop_29_module_wrapper_6_batch_normalization_1_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╧
AssignVariableOp_30AssignVariableOpGassignvariableop_30_module_wrapper_10_batch_normalization_2_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╙
AssignVariableOp_31AssignVariableOpKassignvariableop_31_module_wrapper_10_batch_normalization_2_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╧
AssignVariableOp_32AssignVariableOpGassignvariableop_32_module_wrapper_14_batch_normalization_3_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33╙
AssignVariableOp_33AssignVariableOpKassignvariableop_33_module_wrapper_14_batch_normalization_3_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╧
AssignVariableOp_34AssignVariableOpGassignvariableop_34_module_wrapper_20_batch_normalization_4_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35╙
AssignVariableOp_35AssignVariableOpKassignvariableop_35_module_wrapper_20_batch_normalization_4_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36б
AssignVariableOp_36AssignVariableOpassignvariableop_36_totalIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37б
AssignVariableOp_37AssignVariableOpassignvariableop_37_countIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38г
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39г
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
Identity_42╙
AssignVariableOp_42AssignVariableOpKassignvariableop_42_sgd_module_wrapper_2_batch_normalization_gamma_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43╥
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
Identity_45╟
AssignVariableOp_45AssignVariableOp?assignvariableop_45_sgd_module_wrapper_4_conv2d_1_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46╒
AssignVariableOp_46AssignVariableOpMassignvariableop_46_sgd_module_wrapper_6_batch_normalization_1_gamma_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47╘
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
Identity_49╟
AssignVariableOp_49AssignVariableOp?assignvariableop_49_sgd_module_wrapper_8_conv2d_2_bias_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50╓
AssignVariableOp_50AssignVariableOpNassignvariableop_50_sgd_module_wrapper_10_batch_normalization_2_gamma_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51╒
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
Identity_54╓
AssignVariableOp_54AssignVariableOpNassignvariableop_54_sgd_module_wrapper_14_batch_normalization_3_gamma_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55╒
AssignVariableOp_55AssignVariableOpMassignvariableop_55_sgd_module_wrapper_14_batch_normalization_3_beta_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56╟
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
Identity_58╓
AssignVariableOp_58AssignVariableOpNassignvariableop_58_sgd_module_wrapper_20_batch_normalization_4_gamma_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59╒
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
Identity_61╟
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
Identity_62е
Identity_63IdentityIdentity_62:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_63"#
identity_63Identity_63:output:0*Т
_input_shapesА
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
Л
h
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_95187

args_0
identityf
activation_4/EluEluargs_0*
T0*(
_output_shapes
:         А2
activation_4/Elus
IdentityIdentityactivation_4/Elu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
о
k
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_98147

args_0
identityИw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/ConstТ
dropout_1/dropout/MulMulargs_0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_1/dropout/Mulh
dropout_1/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape╙
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yч
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2 
dropout_1/dropout/GreaterEqualЮ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_1/dropout/Castг
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_1/dropout/Mul_1p
IdentityIdentitydropout_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
р
M
1__inference_module_wrapper_17_layer_call_fn_97970

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
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_952332
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
шv
╗
E__inference_sequential_layer_call_and_return_conditional_losses_96166
module_wrapper_input.
module_wrapper_96077: "
module_wrapper_96079: $
module_wrapper_2_96083: $
module_wrapper_2_96085: $
module_wrapper_2_96087: $
module_wrapper_2_96089: 0
module_wrapper_4_96093: @$
module_wrapper_4_96095:@$
module_wrapper_6_96099:@$
module_wrapper_6_96101:@$
module_wrapper_6_96103:@$
module_wrapper_6_96105:@0
module_wrapper_8_96109:@@$
module_wrapper_8_96111:@%
module_wrapper_10_96115:@%
module_wrapper_10_96117:@%
module_wrapper_10_96119:@%
module_wrapper_10_96121:@2
module_wrapper_12_96125:@А&
module_wrapper_12_96127:	А&
module_wrapper_14_96131:	А&
module_wrapper_14_96133:	А&
module_wrapper_14_96135:	А&
module_wrapper_14_96137:	А+
module_wrapper_18_96143:
АА&
module_wrapper_18_96145:	А&
module_wrapper_20_96149:	А&
module_wrapper_20_96151:	А&
module_wrapper_20_96153:	А&
module_wrapper_20_96155:	А*
module_wrapper_22_96159:	А%
module_wrapper_22_96161:
identityИв&module_wrapper/StatefulPartitionedCallв)module_wrapper_10/StatefulPartitionedCallв)module_wrapper_12/StatefulPartitionedCallв)module_wrapper_14/StatefulPartitionedCallв)module_wrapper_16/StatefulPartitionedCallв)module_wrapper_18/StatefulPartitionedCallв(module_wrapper_2/StatefulPartitionedCallв)module_wrapper_20/StatefulPartitionedCallв)module_wrapper_21/StatefulPartitionedCallв)module_wrapper_22/StatefulPartitionedCallв(module_wrapper_4/StatefulPartitionedCallв(module_wrapper_6/StatefulPartitionedCallв(module_wrapper_8/StatefulPartitionedCall┼
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_96077module_wrapper_96079*
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
GPU 2J 8В *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_956762(
&module_wrapper/StatefulPartitionedCallЬ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_956512"
 module_wrapper_1/PartitionedCallЦ
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0module_wrapper_2_96083module_wrapper_2_96085module_wrapper_2_96087module_wrapper_2_96089*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_956272*
(module_wrapper_2/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_955902"
 module_wrapper_3/PartitionedCallф
(module_wrapper_4/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0module_wrapper_4_96093module_wrapper_4_96095*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_955702*
(module_wrapper_4/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_955452"
 module_wrapper_5/PartitionedCallЦ
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_5/PartitionedCall:output:0module_wrapper_6_96099module_wrapper_6_96101module_wrapper_6_96103module_wrapper_6_96105*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_955212*
(module_wrapper_6/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_954842"
 module_wrapper_7/PartitionedCallф
(module_wrapper_8/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_7/PartitionedCall:output:0module_wrapper_8_96109module_wrapper_8_96111*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_954642*
(module_wrapper_8/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_954392"
 module_wrapper_9/PartitionedCallЭ
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_96115module_wrapper_10_96117module_wrapper_10_96119module_wrapper_10_96121*
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_954152+
)module_wrapper_10/StatefulPartitionedCallв
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_953782#
!module_wrapper_11/PartitionedCallы
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_11/PartitionedCall:output:0module_wrapper_12_96125module_wrapper_12_96127*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_953582+
)module_wrapper_12/StatefulPartitionedCallг
!module_wrapper_13/PartitionedCallPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_953332#
!module_wrapper_13/PartitionedCallЯ
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_13/PartitionedCall:output:0module_wrapper_14_96131module_wrapper_14_96133module_wrapper_14_96135module_wrapper_14_96137*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_953092+
)module_wrapper_14/StatefulPartitionedCallг
!module_wrapper_15/PartitionedCallPartitionedCall2module_wrapper_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_952722#
!module_wrapper_15/PartitionedCall│
)module_wrapper_16/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_952562+
)module_wrapper_16/StatefulPartitionedCallЫ
!module_wrapper_17/PartitionedCallPartitionedCall2module_wrapper_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_952332#
!module_wrapper_17/PartitionedCallу
)module_wrapper_18/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_17/PartitionedCall:output:0module_wrapper_18_96143module_wrapper_18_96145*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_952122+
)module_wrapper_18/StatefulPartitionedCallЫ
!module_wrapper_19/PartitionedCallPartitionedCall2module_wrapper_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_951872#
!module_wrapper_19/PartitionedCallЧ
)module_wrapper_20/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_19/PartitionedCall:output:0module_wrapper_20_96149module_wrapper_20_96151module_wrapper_20_96153module_wrapper_20_96155*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_951632+
)module_wrapper_20/StatefulPartitionedCall▀
)module_wrapper_21/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_20/StatefulPartitionedCall:output:0*^module_wrapper_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_951102+
)module_wrapper_21/StatefulPartitionedCallъ
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_21/StatefulPartitionedCall:output:0module_wrapper_22_96159module_wrapper_22_96161*
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_950832+
)module_wrapper_22/StatefulPartitionedCallЪ
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_950582#
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
Ї

Ш
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_94904

args_08
$dense_matmul_readvariableop_resource:
АА4
%dense_biasadd_readvariableop_resource:	А
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense/MatMul/ReadVariableOpЖ
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/BiasAddи
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
ш
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_97508

args_0
identityо
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
П
и
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_97541

args_0A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@
identityИвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╛
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@*
paddingSAME*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2
conv2d_1/BiasAdd╕
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
б
h
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_97976

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/ConstА
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         А2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
г>
н
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_95163

args_0L
=batch_normalization_4_assignmovingavg_readvariableop_resource:	АN
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:	АA
2batch_normalization_4_cast_readvariableop_resource:	АC
4batch_normalization_4_cast_1_readvariableop_resource:	А
identityИв%batch_normalization_4/AssignMovingAvgв4batch_normalization_4/AssignMovingAvg/ReadVariableOpв'batch_normalization_4/AssignMovingAvg_1в6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpв)batch_normalization_4/Cast/ReadVariableOpв+batch_normalization_4/Cast_1/ReadVariableOp╢
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indices╥
"batch_normalization_4/moments/meanMeanargs_0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2$
"batch_normalization_4/moments/mean┐
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	А2,
*batch_normalization_4/moments/StopGradientч
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceargs_03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:         А21
/batch_normalization_4/moments/SquaredDifference╛
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indicesЛ
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2(
&batch_normalization_4/moments/variance├
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze╦
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1Я
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_4/AssignMovingAvg/decayч
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpё
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/subш
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/mulн
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_4/AssignMovingAvgг
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_4/AssignMovingAvg_1/decayэ
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp∙
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/subЁ
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/mul╖
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_4/AssignMovingAvg_1╞
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp╠
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/y█
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/addж
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/Rsqrt╫
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/mul╣
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_4/batchnorm/mul_1╘
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/mul_2╒
#batch_normalization_4/batchnorm/subSub1batch_normalization_4/Cast/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/sub▐
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_4/batchnorm/add_1Ъ
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
К
Ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96401

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
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
ъ
L
0__inference_module_wrapper_7_layer_call_fn_97643

args_0
identity╤
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_954842
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
К
Ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_96539

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
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
Л
h
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_94915

args_0
identityf
activation_4/EluEluargs_0*
T0*(
_output_shapes
:         А2
activation_4/Elus
IdentityIdentityactivation_4/Elu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
Ё
M
1__inference_module_wrapper_15_layer_call_fn_97923

args_0
identity╙
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_952722
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
Ъ
╟
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_97755

args_0;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identityИв5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1╢
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
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
epsilon%oГ:*
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
я
╘
5__inference_batch_normalization_3_layer_call_fn_98404

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall╡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_966772
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ї

Ш
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_95212

args_08
$dense_matmul_readvariableop_resource:
АА4
%dense_biasadd_readvariableop_resource:	А
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense/MatMul/ReadVariableOpЖ
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/BiasAddи
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
ж
g
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_97566

args_0
identitym
activation_1/EluEluargs_0*
T0*/
_output_shapes
:         !!@2
activation_1/Eluz
IdentityIdentityactivation_1/Elu:activations:0*
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
х
╨
5__inference_batch_normalization_1_layer_call_fn_98293

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCall▓
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
GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_964452
StatefulPartitionedCallи
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
№r
╒
E__inference_sequential_layer_call_and_return_conditional_losses_94978

inputs.
module_wrapper_94658: "
module_wrapper_94660: $
module_wrapper_2_94689: $
module_wrapper_2_94691: $
module_wrapper_2_94693: $
module_wrapper_2_94695: 0
module_wrapper_4_94716: @$
module_wrapper_4_94718:@$
module_wrapper_6_94747:@$
module_wrapper_6_94749:@$
module_wrapper_6_94751:@$
module_wrapper_6_94753:@0
module_wrapper_8_94774:@@$
module_wrapper_8_94776:@%
module_wrapper_10_94805:@%
module_wrapper_10_94807:@%
module_wrapper_10_94809:@%
module_wrapper_10_94811:@2
module_wrapper_12_94832:@А&
module_wrapper_12_94834:	А&
module_wrapper_14_94863:	А&
module_wrapper_14_94865:	А&
module_wrapper_14_94867:	А&
module_wrapper_14_94869:	А+
module_wrapper_18_94905:
АА&
module_wrapper_18_94907:	А&
module_wrapper_20_94938:	А&
module_wrapper_20_94940:	А&
module_wrapper_20_94942:	А&
module_wrapper_20_94944:	А*
module_wrapper_22_94965:	А%
module_wrapper_22_94967:
identityИв&module_wrapper/StatefulPartitionedCallв)module_wrapper_10/StatefulPartitionedCallв)module_wrapper_12/StatefulPartitionedCallв)module_wrapper_14/StatefulPartitionedCallв)module_wrapper_18/StatefulPartitionedCallв(module_wrapper_2/StatefulPartitionedCallв)module_wrapper_20/StatefulPartitionedCallв)module_wrapper_22/StatefulPartitionedCallв(module_wrapper_4/StatefulPartitionedCallв(module_wrapper_6/StatefulPartitionedCallв(module_wrapper_8/StatefulPartitionedCall╖
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_94658module_wrapper_94660*
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
GPU 2J 8В *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_946572(
&module_wrapper/StatefulPartitionedCallЬ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_946682"
 module_wrapper_1/PartitionedCallШ
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0module_wrapper_2_94689module_wrapper_2_94691module_wrapper_2_94693module_wrapper_2_94695*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_946882*
(module_wrapper_2/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_947032"
 module_wrapper_3/PartitionedCallф
(module_wrapper_4/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0module_wrapper_4_94716module_wrapper_4_94718*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_947152*
(module_wrapper_4/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_947262"
 module_wrapper_5/PartitionedCallШ
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_5/PartitionedCall:output:0module_wrapper_6_94747module_wrapper_6_94749module_wrapper_6_94751module_wrapper_6_94753*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_947462*
(module_wrapper_6/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_947612"
 module_wrapper_7/PartitionedCallф
(module_wrapper_8/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_7/PartitionedCall:output:0module_wrapper_8_94774module_wrapper_8_94776*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_947732*
(module_wrapper_8/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_947842"
 module_wrapper_9/PartitionedCallЯ
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_94805module_wrapper_10_94807module_wrapper_10_94809module_wrapper_10_94811*
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_948042+
)module_wrapper_10/StatefulPartitionedCallв
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_948192#
!module_wrapper_11/PartitionedCallы
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_11/PartitionedCall:output:0module_wrapper_12_94832module_wrapper_12_94834*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_948312+
)module_wrapper_12/StatefulPartitionedCallг
!module_wrapper_13/PartitionedCallPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_948422#
!module_wrapper_13/PartitionedCallб
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_13/PartitionedCall:output:0module_wrapper_14_94863module_wrapper_14_94865module_wrapper_14_94867module_wrapper_14_94869*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_948622+
)module_wrapper_14/StatefulPartitionedCallг
!module_wrapper_15/PartitionedCallPartitionedCall2module_wrapper_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_948772#
!module_wrapper_15/PartitionedCallЫ
!module_wrapper_16/PartitionedCallPartitionedCall*module_wrapper_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_948842#
!module_wrapper_16/PartitionedCallУ
!module_wrapper_17/PartitionedCallPartitionedCall*module_wrapper_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_948922#
!module_wrapper_17/PartitionedCallу
)module_wrapper_18/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_17/PartitionedCall:output:0module_wrapper_18_94905module_wrapper_18_94907*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_949042+
)module_wrapper_18/StatefulPartitionedCallЫ
!module_wrapper_19/PartitionedCallPartitionedCall2module_wrapper_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_949152#
!module_wrapper_19/PartitionedCallЩ
)module_wrapper_20/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_19/PartitionedCall:output:0module_wrapper_20_94938module_wrapper_20_94940module_wrapper_20_94942module_wrapper_20_94944*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_949372+
)module_wrapper_20/StatefulPartitionedCallЫ
!module_wrapper_21/PartitionedCallPartitionedCall2module_wrapper_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_949522#
!module_wrapper_21/PartitionedCallт
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_21/PartitionedCall:output:0module_wrapper_22_94965module_wrapper_22_94967*
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_949642+
)module_wrapper_22/StatefulPartitionedCallЪ
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_949752#
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
є
h
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_97933

args_0
identity│
max_pooling2d_3/MaxPoolMaxPoolargs_0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool}
IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
я
h
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_97788

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
▓
б
1__inference_module_wrapper_18_layer_call_fn_97991

args_0
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_949042
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
Ї

Ш
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_98020

args_08
$dense_matmul_readvariableop_resource:
АА4
%dense_biasadd_readvariableop_resource:	А
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense/MatMul/ReadVariableOpЖ
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/BiasAddи
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
я
h
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_95378

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
У
h
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_95058

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
╬
├
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_96721

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Х
╠
1__inference_module_wrapper_10_layer_call_fn_97737

args_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЬ
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_954152
StatefulPartitionedCallЦ
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
Я
╨
1__inference_module_wrapper_14_layer_call_fn_97864

args_0
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_948622
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
л
h
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_94884

args_0
identitys
dropout/IdentityIdentityargs_0*
T0*0
_output_shapes
:         А2
dropout/Identityv
IdentityIdentitydropout/Identity:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
ъ
Ы
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_95309

args_0<
-batch_normalization_3_readvariableop_resource:	А>
/batch_normalization_3_readvariableop_1_resource:	АM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв$batch_normalization_3/AssignNewValueв&batch_normalization_3/AssignNewValue_1в5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1╖
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization_3/ReadVariableOp╜
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&batch_normalization_3/ReadVariableOp_1ъ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpЁ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1с
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
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
&batch_normalization_3/AssignNewValue_1Щ
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
ж
g
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_94726

args_0
identitym
activation_1/EluEluargs_0*
T0*/
_output_shapes
:         !!@2
activation_1/Eluz
IdentityIdentityactivation_1/Elu:activations:0*
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
╫
и
1__inference_module_wrapper_12_layer_call_fn_97802

args_0"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_948312
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

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
л
h
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_97851

args_0
identityn
activation_3/EluEluargs_0*
T0*0
_output_shapes
:         А2
activation_3/Elu{
IdentityIdentityactivation_3/Elu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
╨
M
1__inference_module_wrapper_19_layer_call_fn_98030

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
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_951872
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
л
h
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_97948

args_0
identitys
dropout/IdentityIdentityargs_0*
T0*0
_output_shapes
:         А2
dropout/Identityv
IdentityIdentitydropout/Identity:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
ж
│
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_98086

args_0A
2batch_normalization_4_cast_readvariableop_resource:	АC
4batch_normalization_4_cast_1_readvariableop_resource:	АC
4batch_normalization_4_cast_2_readvariableop_resource:	АC
4batch_normalization_4_cast_3_readvariableop_resource:	А
identityИв)batch_normalization_4/Cast/ReadVariableOpв+batch_normalization_4/Cast_1/ReadVariableOpв+batch_normalization_4/Cast_2/ReadVariableOpв+batch_normalization_4/Cast_3/ReadVariableOp╞
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp╠
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOp╠
+batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+batch_normalization_4/Cast_2/ReadVariableOp╠
+batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+batch_normalization_4/Cast_3/ReadVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/y▐
#batch_normalization_4/batchnorm/addAddV23batch_normalization_4/Cast_1/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/addж
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/Rsqrt╫
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/mul╣
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_4/batchnorm/mul_1╫
%batch_normalization_4/batchnorm/mul_2Mul1batch_normalization_4/Cast/ReadVariableOp:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/mul_2╫
#batch_normalization_4/batchnorm/subSub3batch_normalization_4/Cast_2/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/sub▐
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_4/batchnorm/add_1┤
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp,^batch_normalization_4/Cast_2/ReadVariableOp,^batch_normalization_4/Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2Z
+batch_normalization_4/Cast_2/ReadVariableOp+batch_normalization_4/Cast_2/ReadVariableOp2Z
+batch_normalization_4/Cast_3/ReadVariableOp+batch_normalization_4/Cast_3/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
Э
Ю
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_98185

args_09
&dense_1_matmul_readvariableop_resource:	А5
'dense_1_biasadd_readvariableop_resource:
identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpж
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_1/MatMul/ReadVariableOpЛ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddн
IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
о
Я
1__inference_module_wrapper_22_layer_call_fn_98156

args_0
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall№
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_949642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
Ъ
╟
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_94804

args_0;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identityИв5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1╢
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
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
epsilon%oГ:*
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
Ё
M
1__inference_module_wrapper_16_layer_call_fn_97938

args_0
identity╙
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_948842
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
№
j
1__inference_module_wrapper_16_layer_call_fn_97943

args_0
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_952562
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
Э
Ю
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_94964

args_09
&dense_1_matmul_readvariableop_resource:	А5
'dense_1_biasadd_readvariableop_resource:
identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpж
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_1/MatMul/ReadVariableOpЛ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddн
IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
р
M
1__inference_module_wrapper_17_layer_call_fn_97965

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
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_948922
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
ю
g
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_95484

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
╛v
н
E__inference_sequential_layer_call_and_return_conditional_losses_95846

inputs.
module_wrapper_95757: "
module_wrapper_95759: $
module_wrapper_2_95763: $
module_wrapper_2_95765: $
module_wrapper_2_95767: $
module_wrapper_2_95769: 0
module_wrapper_4_95773: @$
module_wrapper_4_95775:@$
module_wrapper_6_95779:@$
module_wrapper_6_95781:@$
module_wrapper_6_95783:@$
module_wrapper_6_95785:@0
module_wrapper_8_95789:@@$
module_wrapper_8_95791:@%
module_wrapper_10_95795:@%
module_wrapper_10_95797:@%
module_wrapper_10_95799:@%
module_wrapper_10_95801:@2
module_wrapper_12_95805:@А&
module_wrapper_12_95807:	А&
module_wrapper_14_95811:	А&
module_wrapper_14_95813:	А&
module_wrapper_14_95815:	А&
module_wrapper_14_95817:	А+
module_wrapper_18_95823:
АА&
module_wrapper_18_95825:	А&
module_wrapper_20_95829:	А&
module_wrapper_20_95831:	А&
module_wrapper_20_95833:	А&
module_wrapper_20_95835:	А*
module_wrapper_22_95839:	А%
module_wrapper_22_95841:
identityИв&module_wrapper/StatefulPartitionedCallв)module_wrapper_10/StatefulPartitionedCallв)module_wrapper_12/StatefulPartitionedCallв)module_wrapper_14/StatefulPartitionedCallв)module_wrapper_16/StatefulPartitionedCallв)module_wrapper_18/StatefulPartitionedCallв(module_wrapper_2/StatefulPartitionedCallв)module_wrapper_20/StatefulPartitionedCallв)module_wrapper_21/StatefulPartitionedCallв)module_wrapper_22/StatefulPartitionedCallв(module_wrapper_4/StatefulPartitionedCallв(module_wrapper_6/StatefulPartitionedCallв(module_wrapper_8/StatefulPartitionedCall╖
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_95757module_wrapper_95759*
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
GPU 2J 8В *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_956762(
&module_wrapper/StatefulPartitionedCallЬ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_956512"
 module_wrapper_1/PartitionedCallЦ
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0module_wrapper_2_95763module_wrapper_2_95765module_wrapper_2_95767module_wrapper_2_95769*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_956272*
(module_wrapper_2/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_955902"
 module_wrapper_3/PartitionedCallф
(module_wrapper_4/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0module_wrapper_4_95773module_wrapper_4_95775*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_955702*
(module_wrapper_4/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_955452"
 module_wrapper_5/PartitionedCallЦ
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_5/PartitionedCall:output:0module_wrapper_6_95779module_wrapper_6_95781module_wrapper_6_95783module_wrapper_6_95785*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_955212*
(module_wrapper_6/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_954842"
 module_wrapper_7/PartitionedCallф
(module_wrapper_8/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_7/PartitionedCall:output:0module_wrapper_8_95789module_wrapper_8_95791*
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_954642*
(module_wrapper_8/StatefulPartitionedCallЮ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_954392"
 module_wrapper_9/PartitionedCallЭ
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_95795module_wrapper_10_95797module_wrapper_10_95799module_wrapper_10_95801*
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_954152+
)module_wrapper_10/StatefulPartitionedCallв
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_953782#
!module_wrapper_11/PartitionedCallы
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_11/PartitionedCall:output:0module_wrapper_12_95805module_wrapper_12_95807*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_953582+
)module_wrapper_12/StatefulPartitionedCallг
!module_wrapper_13/PartitionedCallPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_953332#
!module_wrapper_13/PartitionedCallЯ
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_13/PartitionedCall:output:0module_wrapper_14_95811module_wrapper_14_95813module_wrapper_14_95815module_wrapper_14_95817*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_953092+
)module_wrapper_14/StatefulPartitionedCallг
!module_wrapper_15/PartitionedCallPartitionedCall2module_wrapper_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_952722#
!module_wrapper_15/PartitionedCall│
)module_wrapper_16/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_952562+
)module_wrapper_16/StatefulPartitionedCallЫ
!module_wrapper_17/PartitionedCallPartitionedCall2module_wrapper_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_952332#
!module_wrapper_17/PartitionedCallу
)module_wrapper_18/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_17/PartitionedCall:output:0module_wrapper_18_95823module_wrapper_18_95825*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_952122+
)module_wrapper_18/StatefulPartitionedCallЫ
!module_wrapper_19/PartitionedCallPartitionedCall2module_wrapper_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_951872#
!module_wrapper_19/PartitionedCallЧ
)module_wrapper_20/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_19/PartitionedCall:output:0module_wrapper_20_95829module_wrapper_20_95831module_wrapper_20_95833module_wrapper_20_95835*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_951632+
)module_wrapper_20/StatefulPartitionedCall▀
)module_wrapper_21/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_20/StatefulPartitionedCall:output:0*^module_wrapper_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_951102+
)module_wrapper_21/StatefulPartitionedCallъ
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_21/StatefulPartitionedCall:output:0module_wrapper_22_95839module_wrapper_22_95841*
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_950832+
)module_wrapper_22/StatefulPartitionedCallЪ
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_950582#
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
ч
╨
5__inference_batch_normalization_1_layer_call_fn_98280

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCall┤
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
GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_964012
StatefulPartitionedCallи
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
Е
╘
5__inference_batch_normalization_4_layer_call_fn_98479

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_968772
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╛
┐
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96445

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1Р
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
є
h
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_97928

args_0
identity│
max_pooling2d_3/MaxPoolMaxPoolargs_0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool}
IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
╠
M
1__inference_module_wrapper_23_layer_call_fn_98190

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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_949752
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
л
h
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_94842

args_0
identityn
activation_3/EluEluargs_0*
T0*0
_output_shapes
:         А2
activation_3/Elu{
IdentityIdentityactivation_3/Elu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
ж
g
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_97706

args_0
identitym
activation_2/EluEluargs_0*
T0*/
_output_shapes
:         @2
activation_2/Eluz
IdentityIdentityactivation_2/Elu:activations:0*
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
с
╬
3__inference_batch_normalization_layer_call_fn_98231

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCall░
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
GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_963072
StatefulPartitionedCallи
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
╤
I
-__inference_max_pooling2d_layer_call_fn_96379

inputs
identityщ
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
GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_963732
PartitionedCallП
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
И
Щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_98249

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
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
К
Ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_98373

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
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
┘
Ц
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_95521

args_0;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identityИв$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
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
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
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
&batch_normalization_1/AssignNewValue_1Ш
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
ь
M
1__inference_module_wrapper_11_layer_call_fn_97778

args_0
identity╥
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_948192
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
 
╨
1__inference_module_wrapper_20_layer_call_fn_98053

args_0
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_949372
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
═
г
.__inference_module_wrapper_layer_call_fn_97382

args_0!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallБ
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
GPU 2J 8В *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_946572
StatefulPartitionedCallЦ
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
П
и
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_97681

args_0A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@
identityИвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOp░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp╛
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_2/Conv2Dз
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpм
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAdd╕
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
╤
е
0__inference_module_wrapper_8_layer_call_fn_97662

args_0!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallГ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_947732
StatefulPartitionedCallЦ
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
П
и
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_94773

args_0A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@
identityИвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOp░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp╛
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_2/Conv2Dз
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpм
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAdd╕
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
в)
╒
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_98533

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А+
cast_readvariableop_resource:	А-
cast_1_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвCast/ReadVariableOpвCast_1/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Д
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1А
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╔
▄
*__inference_sequential_layer_call_fn_97024

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

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23:
АА

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:
identityИвStatefulPartitionedCallР
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
GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_949782
StatefulPartitionedCallО
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
к
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_96649

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
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
╬
├
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_98453

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ч
╠
1__inference_module_wrapper_10_layer_call_fn_97724

args_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЮ
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_948042
StatefulPartitionedCallЦ
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
╠
M
1__inference_module_wrapper_23_layer_call_fn_98195

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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_950582
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
▓ф
╛$
E__inference_sequential_layer_call_and_return_conditional_losses_97219

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
9module_wrapper_12_conv2d_3_conv2d_readvariableop_resource:@АI
:module_wrapper_12_conv2d_3_biasadd_readvariableop_resource:	АN
?module_wrapper_14_batch_normalization_3_readvariableop_resource:	АP
Amodule_wrapper_14_batch_normalization_3_readvariableop_1_resource:	А_
Pmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	Аa
Rmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АJ
6module_wrapper_18_dense_matmul_readvariableop_resource:
ААF
7module_wrapper_18_dense_biasadd_readvariableop_resource:	АS
Dmodule_wrapper_20_batch_normalization_4_cast_readvariableop_resource:	АU
Fmodule_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource:	АU
Fmodule_wrapper_20_batch_normalization_4_cast_2_readvariableop_resource:	АU
Fmodule_wrapper_20_batch_normalization_4_cast_3_readvariableop_resource:	АK
8module_wrapper_22_dense_1_matmul_readvariableop_resource:	АG
9module_wrapper_22_dense_1_biasadd_readvariableop_resource:
identityИв,module_wrapper/conv2d/BiasAdd/ReadVariableOpв+module_wrapper/conv2d/Conv2D/ReadVariableOpвGmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвImodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в6module_wrapper_10/batch_normalization_2/ReadVariableOpв8module_wrapper_10/batch_normalization_2/ReadVariableOp_1в1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpв0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpвGmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвImodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в6module_wrapper_14/batch_normalization_3/ReadVariableOpв8module_wrapper_14/batch_normalization_3/ReadVariableOp_1в.module_wrapper_18/dense/BiasAdd/ReadVariableOpв-module_wrapper_18/dense/MatMul/ReadVariableOpвDmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpвFmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1в3module_wrapper_2/batch_normalization/ReadVariableOpв5module_wrapper_2/batch_normalization/ReadVariableOp_1в;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpв=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpв=module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpв=module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpв0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpв/module_wrapper_22/dense_1/MatMul/ReadVariableOpв0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpв/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpвFmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpвHmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в5module_wrapper_6/batch_normalization_1/ReadVariableOpв7module_wrapper_6/batch_normalization_1/ReadVariableOp_1в0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpв/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp╫
+module_wrapper/conv2d/Conv2D/ReadVariableOpReadVariableOp4module_wrapper_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+module_wrapper/conv2d/Conv2D/ReadVariableOpх
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
,module_wrapper/conv2d/BiasAdd/ReadVariableOpр
module_wrapper/conv2d/BiasAddBiasAdd%module_wrapper/conv2d/Conv2D:output:04module_wrapper/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd 2
module_wrapper/conv2d/BiasAddл
module_wrapper_1/activation/EluElu&module_wrapper/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         dd 2!
module_wrapper_1/activation/Eluу
3module_wrapper_2/batch_normalization/ReadVariableOpReadVariableOp<module_wrapper_2_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype025
3module_wrapper_2/batch_normalization/ReadVariableOpщ
5module_wrapper_2/batch_normalization/ReadVariableOp_1ReadVariableOp>module_wrapper_2_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype027
5module_wrapper_2/batch_normalization/ReadVariableOp_1Ц
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpЬ
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1╧
5module_wrapper_2/batch_normalization/FusedBatchNormV3FusedBatchNormV3-module_wrapper_1/activation/Elu:activations:0;module_wrapper_2/batch_normalization/ReadVariableOp:value:0=module_wrapper_2/batch_normalization/ReadVariableOp_1:value:0Lmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Nmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oГ:*
is_training( 27
5module_wrapper_2/batch_normalization/FusedBatchNormV3Г
&module_wrapper_3/max_pooling2d/MaxPoolMaxPool9module_wrapper_2/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2(
&module_wrapper_3/max_pooling2d/MaxPoolу
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_4_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype021
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpЪ
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
0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpЁ
!module_wrapper_4/conv2d_1/BiasAddBiasAdd)module_wrapper_4/conv2d_1/Conv2D:output:08module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2#
!module_wrapper_4/conv2d_1/BiasAdd│
!module_wrapper_5/activation_1/EluElu*module_wrapper_4/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         !!@2#
!module_wrapper_5/activation_1/Eluщ
5module_wrapper_6/batch_normalization_1/ReadVariableOpReadVariableOp>module_wrapper_6_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype027
5module_wrapper_6/batch_normalization_1/ReadVariableOpя
7module_wrapper_6/batch_normalization_1/ReadVariableOp_1ReadVariableOp@module_wrapper_6_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7module_wrapper_6/batch_normalization_1/ReadVariableOp_1Ь
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02H
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpв
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1▌
7module_wrapper_6/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3/module_wrapper_5/activation_1/Elu:activations:0=module_wrapper_6/batch_normalization_1/ReadVariableOp:value:0?module_wrapper_6/batch_normalization_1/ReadVariableOp_1:value:0Nmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Pmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !!@:@:@:@:@:*
epsilon%oГ:*
is_training( 29
7module_wrapper_6/batch_normalization_1/FusedBatchNormV3Й
(module_wrapper_7/max_pooling2d_1/MaxPoolMaxPool;module_wrapper_6/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2*
(module_wrapper_7/max_pooling2d_1/MaxPoolу
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_8_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype021
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpЬ
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
0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpЁ
!module_wrapper_8/conv2d_2/BiasAddBiasAdd)module_wrapper_8/conv2d_2/Conv2D:output:08module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2#
!module_wrapper_8/conv2d_2/BiasAdd│
!module_wrapper_9/activation_2/EluElu*module_wrapper_8/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2#
!module_wrapper_9/activation_2/Eluь
6module_wrapper_10/batch_normalization_2/ReadVariableOpReadVariableOp?module_wrapper_10_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype028
6module_wrapper_10/batch_normalization_2/ReadVariableOpЄ
8module_wrapper_10/batch_normalization_2/ReadVariableOp_1ReadVariableOpAmodule_wrapper_10_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8module_wrapper_10/batch_normalization_2/ReadVariableOp_1Я
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpе
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1у
8module_wrapper_10/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3/module_wrapper_9/activation_2/Elu:activations:0>module_wrapper_10/batch_normalization_2/ReadVariableOp:value:0@module_wrapper_10/batch_normalization_2/ReadVariableOp_1:value:0Omodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Qmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2:
8module_wrapper_10/batch_normalization_2/FusedBatchNormV3М
)module_wrapper_11/max_pooling2d_2/MaxPoolMaxPool<module_wrapper_10/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2+
)module_wrapper_11/max_pooling2d_2/MaxPoolч
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpReadVariableOp9module_wrapper_12_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype022
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpб
!module_wrapper_12/conv2d_3/Conv2DConv2D2module_wrapper_11/max_pooling2d_2/MaxPool:output:08module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2#
!module_wrapper_12/conv2d_3/Conv2D▐
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp:module_wrapper_12_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype023
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpї
"module_wrapper_12/conv2d_3/BiasAddBiasAdd*module_wrapper_12/conv2d_3/Conv2D:output:09module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2$
"module_wrapper_12/conv2d_3/BiasAdd╖
"module_wrapper_13/activation_3/EluElu+module_wrapper_12/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:         А2$
"module_wrapper_13/activation_3/Eluэ
6module_wrapper_14/batch_normalization_3/ReadVariableOpReadVariableOp?module_wrapper_14_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6module_wrapper_14/batch_normalization_3/ReadVariableOpє
8module_wrapper_14/batch_normalization_3/ReadVariableOp_1ReadVariableOpAmodule_wrapper_14_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8module_wrapper_14/batch_normalization_3/ReadVariableOp_1а
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02I
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpж
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02K
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1щ
8module_wrapper_14/batch_normalization_3/FusedBatchNormV3FusedBatchNormV30module_wrapper_13/activation_3/Elu:activations:0>module_wrapper_14/batch_normalization_3/ReadVariableOp:value:0@module_wrapper_14/batch_normalization_3/ReadVariableOp_1:value:0Omodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Qmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2:
8module_wrapper_14/batch_normalization_3/FusedBatchNormV3Н
)module_wrapper_15/max_pooling2d_3/MaxPoolMaxPool<module_wrapper_14/batch_normalization_3/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2+
)module_wrapper_15/max_pooling2d_3/MaxPool├
"module_wrapper_16/dropout/IdentityIdentity2module_wrapper_15/max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:         А2$
"module_wrapper_16/dropout/IdentityУ
module_wrapper_17/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
module_wrapper_17/flatten/Const█
!module_wrapper_17/flatten/ReshapeReshape+module_wrapper_16/dropout/Identity:output:0(module_wrapper_17/flatten/Const:output:0*
T0*(
_output_shapes
:         А2#
!module_wrapper_17/flatten/Reshape╫
-module_wrapper_18/dense/MatMul/ReadVariableOpReadVariableOp6module_wrapper_18_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02/
-module_wrapper_18/dense/MatMul/ReadVariableOpр
module_wrapper_18/dense/MatMulMatMul*module_wrapper_17/flatten/Reshape:output:05module_wrapper_18/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
module_wrapper_18/dense/MatMul╒
.module_wrapper_18/dense/BiasAdd/ReadVariableOpReadVariableOp7module_wrapper_18_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.module_wrapper_18/dense/BiasAdd/ReadVariableOpт
module_wrapper_18/dense/BiasAddBiasAdd(module_wrapper_18/dense/MatMul:product:06module_wrapper_18/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
module_wrapper_18/dense/BiasAddм
"module_wrapper_19/activation_4/EluElu(module_wrapper_18/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2$
"module_wrapper_19/activation_4/Elu№
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpReadVariableOpDmodule_wrapper_20_batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpВ
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpReadVariableOpFmodule_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02?
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpВ
=module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpReadVariableOpFmodule_wrapper_20_batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:А*
dtype02?
=module_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOpВ
=module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOpReadVariableOpFmodule_wrapper_20_batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:А*
dtype02?
=module_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp╖
7module_wrapper_20/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7module_wrapper_20/batch_normalization_4/batchnorm/add/yж
5module_wrapper_20/batch_normalization_4/batchnorm/addAddV2Emodule_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp:value:0@module_wrapper_20/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А27
5module_wrapper_20/batch_normalization_4/batchnorm/add▄
7module_wrapper_20/batch_normalization_4/batchnorm/RsqrtRsqrt9module_wrapper_20/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А29
7module_wrapper_20/batch_normalization_4/batchnorm/RsqrtЯ
5module_wrapper_20/batch_normalization_4/batchnorm/mulMul;module_wrapper_20/batch_normalization_4/batchnorm/Rsqrt:y:0Emodule_wrapper_20/batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:А27
5module_wrapper_20/batch_normalization_4/batchnorm/mulЩ
7module_wrapper_20/batch_normalization_4/batchnorm/mul_1Mul0module_wrapper_19/activation_4/Elu:activations:09module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А29
7module_wrapper_20/batch_normalization_4/batchnorm/mul_1Я
7module_wrapper_20/batch_normalization_4/batchnorm/mul_2MulCmodule_wrapper_20/batch_normalization_4/Cast/ReadVariableOp:value:09module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А29
7module_wrapper_20/batch_normalization_4/batchnorm/mul_2Я
5module_wrapper_20/batch_normalization_4/batchnorm/subSubEmodule_wrapper_20/batch_normalization_4/Cast_2/ReadVariableOp:value:0;module_wrapper_20/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А27
5module_wrapper_20/batch_normalization_4/batchnorm/subж
7module_wrapper_20/batch_normalization_4/batchnorm/add_1AddV2;module_wrapper_20/batch_normalization_4/batchnorm/mul_1:z:09module_wrapper_20/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А29
7module_wrapper_20/batch_normalization_4/batchnorm/add_1╚
$module_wrapper_21/dropout_1/IdentityIdentity;module_wrapper_20/batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         А2&
$module_wrapper_21/dropout_1/Identity▄
/module_wrapper_22/dense_1/MatMul/ReadVariableOpReadVariableOp8module_wrapper_22_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype021
/module_wrapper_22/dense_1/MatMul/ReadVariableOpш
 module_wrapper_22/dense_1/MatMulMatMul-module_wrapper_21/dropout_1/Identity:output:07module_wrapper_22/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 module_wrapper_22/dense_1/MatMul┌
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_22_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpщ
!module_wrapper_22/dense_1/BiasAddBiasAdd*module_wrapper_22/dense_1/MatMul:product:08module_wrapper_22/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2#
!module_wrapper_22/dense_1/BiasAdd╣
&module_wrapper_23/activation_5/SoftmaxSoftmax*module_wrapper_22/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2(
&module_wrapper_23/activation_5/Softmaxю
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
+module_wrapper/conv2d/Conv2D/ReadVariableOp+module_wrapper/conv2d/Conv2D/ReadVariableOp2Т
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpGmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2Ц
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12p
6module_wrapper_10/batch_normalization_2/ReadVariableOp6module_wrapper_10/batch_normalization_2/ReadVariableOp2t
8module_wrapper_10/batch_normalization_2/ReadVariableOp_18module_wrapper_10/batch_normalization_2/ReadVariableOp_12f
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp2d
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp2Т
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpGmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2Ц
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12p
6module_wrapper_14/batch_normalization_3/ReadVariableOp6module_wrapper_14/batch_normalization_3/ReadVariableOp2t
8module_wrapper_14/batch_normalization_3/ReadVariableOp_18module_wrapper_14/batch_normalization_3/ReadVariableOp_12`
.module_wrapper_18/dense/BiasAdd/ReadVariableOp.module_wrapper_18/dense/BiasAdd/ReadVariableOp2^
-module_wrapper_18/dense/MatMul/ReadVariableOp-module_wrapper_18/dense/MatMul/ReadVariableOp2М
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpDmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp2Р
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
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp2Р
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpFmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2Ф
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5module_wrapper_6/batch_normalization_1/ReadVariableOp5module_wrapper_6/batch_normalization_1/ReadVariableOp2r
7module_wrapper_6/batch_normalization_1/ReadVariableOp_17module_wrapper_6/batch_normalization_1/ReadVariableOp_12d
0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp2b
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
б
h
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_94892

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/ConstА
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         А2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
╒
K
/__inference_max_pooling2d_2_layer_call_fn_96655

inputs
identityы
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
GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_966492
PartitionedCallП
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
ю
g
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_97648

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
ц
З
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_98499

inputs+
cast_readvariableop_resource:	А-
cast_1_readvariableop_resource:	А-
cast_2_readvariableop_resource:	А-
cast_3_readvariableop_resource:	А
identityИвCast/ReadVariableOpвCast_1/ReadVariableOpвCast_2/ReadVariableOpвCast_3/ReadVariableOpД
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast_1/ReadVariableOpК
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast_2/ReadVariableOpК
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:А*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЖ
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1╞
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
а
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_97431

args_0
identityi
activation/EluEluargs_0*
T0*/
_output_shapes
:         dd 2
activation/Elux
IdentityIdentityactivation/Elu:activations:0*
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
г>
н
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_98120

args_0L
=batch_normalization_4_assignmovingavg_readvariableop_resource:	АN
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:	АA
2batch_normalization_4_cast_readvariableop_resource:	АC
4batch_normalization_4_cast_1_readvariableop_resource:	А
identityИв%batch_normalization_4/AssignMovingAvgв4batch_normalization_4/AssignMovingAvg/ReadVariableOpв'batch_normalization_4/AssignMovingAvg_1в6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpв)batch_normalization_4/Cast/ReadVariableOpв+batch_normalization_4/Cast_1/ReadVariableOp╢
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indices╥
"batch_normalization_4/moments/meanMeanargs_0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2$
"batch_normalization_4/moments/mean┐
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	А2,
*batch_normalization_4/moments/StopGradientч
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceargs_03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:         А21
/batch_normalization_4/moments/SquaredDifference╛
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indicesЛ
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2(
&batch_normalization_4/moments/variance├
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze╦
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1Я
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_4/AssignMovingAvg/decayч
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpё
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/subш
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/mulн
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_4/AssignMovingAvgг
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_4/AssignMovingAvg_1/decayэ
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp∙
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/subЁ
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/mul╖
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_4/AssignMovingAvg_1╞
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp╠
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/y█
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/addж
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/Rsqrt╫
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/mul╣
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_4/batchnorm/mul_1╘
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/mul_2╒
#batch_normalization_4/batchnorm/subSub1batch_normalization_4/Cast/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/sub▐
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_4/batchnorm/add_1Ъ
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
к
╦
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_94862

args_0<
-batch_normalization_3_readvariableop_resource:	А>
/batch_normalization_3_readvariableop_1_resource:	АM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1╖
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization_3/ReadVariableOp╜
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&batch_normalization_3/ReadVariableOp_1ъ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpЁ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1╙
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3args_0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3╔
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
╒
K
/__inference_max_pooling2d_1_layer_call_fn_96517

inputs
identityы
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
GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_965112
PartitionedCallП
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
є
ъ
*__inference_sequential_layer_call_fn_95045
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

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23:
АА

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:
identityИвStatefulPartitionedCallЮ
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
GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_949782
StatefulPartitionedCallО
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
ш
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_94703

args_0
identityо
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
Ч
л
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_94831

args_0B
'conv2d_3_conv2d_readvariableop_resource:@А7
(conv2d_3_biasadd_readvariableop_resource:	А
identityИвconv2d_3/BiasAdd/ReadVariableOpвconv2d_3/Conv2D/ReadVariableOp▒
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_3/Conv2D/ReadVariableOp┐
conv2d_3/Conv2DConv2Dargs_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_3/Conv2Dи
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpн
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_3/BiasAdd╣
IdentityIdentityconv2d_3/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

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
л
h
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_95333

args_0
identityn
activation_3/EluEluargs_0*
T0*0
_output_shapes
:         А2
activation_3/Elu{
IdentityIdentityactivation_3/Elu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
ю
g
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_97653

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
╦
╢
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_94688

args_09
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: 
identityИв3batch_normalization/FusedBatchNormV3/ReadVariableOpв5batch_normalization/FusedBatchNormV3/ReadVariableOp_1в"batch_normalization/ReadVariableOpв$batch_normalization/ReadVariableOp_1░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
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
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3╛
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
╨
M
1__inference_module_wrapper_19_layer_call_fn_98025

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
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_949152
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
Ш╬
д)
E__inference_sequential_layer_call_and_return_conditional_losses_97373

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
9module_wrapper_12_conv2d_3_conv2d_readvariableop_resource:@АI
:module_wrapper_12_conv2d_3_biasadd_readvariableop_resource:	АN
?module_wrapper_14_batch_normalization_3_readvariableop_resource:	АP
Amodule_wrapper_14_batch_normalization_3_readvariableop_1_resource:	А_
Pmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	Аa
Rmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АJ
6module_wrapper_18_dense_matmul_readvariableop_resource:
ААF
7module_wrapper_18_dense_biasadd_readvariableop_resource:	А^
Omodule_wrapper_20_batch_normalization_4_assignmovingavg_readvariableop_resource:	А`
Qmodule_wrapper_20_batch_normalization_4_assignmovingavg_1_readvariableop_resource:	АS
Dmodule_wrapper_20_batch_normalization_4_cast_readvariableop_resource:	АU
Fmodule_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource:	АK
8module_wrapper_22_dense_1_matmul_readvariableop_resource:	АG
9module_wrapper_22_dense_1_biasadd_readvariableop_resource:
identityИв,module_wrapper/conv2d/BiasAdd/ReadVariableOpв+module_wrapper/conv2d/Conv2D/ReadVariableOpв6module_wrapper_10/batch_normalization_2/AssignNewValueв8module_wrapper_10/batch_normalization_2/AssignNewValue_1вGmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвImodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в6module_wrapper_10/batch_normalization_2/ReadVariableOpв8module_wrapper_10/batch_normalization_2/ReadVariableOp_1в1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpв0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpв6module_wrapper_14/batch_normalization_3/AssignNewValueв8module_wrapper_14/batch_normalization_3/AssignNewValue_1вGmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвImodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в6module_wrapper_14/batch_normalization_3/ReadVariableOpв8module_wrapper_14/batch_normalization_3/ReadVariableOp_1в.module_wrapper_18/dense/BiasAdd/ReadVariableOpв-module_wrapper_18/dense/MatMul/ReadVariableOpв3module_wrapper_2/batch_normalization/AssignNewValueв5module_wrapper_2/batch_normalization/AssignNewValue_1вDmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpвFmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1в3module_wrapper_2/batch_normalization/ReadVariableOpв5module_wrapper_2/batch_normalization/ReadVariableOp_1в7module_wrapper_20/batch_normalization_4/AssignMovingAvgвFmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOpв9module_wrapper_20/batch_normalization_4/AssignMovingAvg_1вHmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpв;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpв=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpв0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpв/module_wrapper_22/dense_1/MatMul/ReadVariableOpв0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpв/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpв5module_wrapper_6/batch_normalization_1/AssignNewValueв7module_wrapper_6/batch_normalization_1/AssignNewValue_1вFmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpвHmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в5module_wrapper_6/batch_normalization_1/ReadVariableOpв7module_wrapper_6/batch_normalization_1/ReadVariableOp_1в0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpв/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp╫
+module_wrapper/conv2d/Conv2D/ReadVariableOpReadVariableOp4module_wrapper_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+module_wrapper/conv2d/Conv2D/ReadVariableOpх
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
,module_wrapper/conv2d/BiasAdd/ReadVariableOpр
module_wrapper/conv2d/BiasAddBiasAdd%module_wrapper/conv2d/Conv2D:output:04module_wrapper/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd 2
module_wrapper/conv2d/BiasAddл
module_wrapper_1/activation/EluElu&module_wrapper/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         dd 2!
module_wrapper_1/activation/Eluу
3module_wrapper_2/batch_normalization/ReadVariableOpReadVariableOp<module_wrapper_2_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype025
3module_wrapper_2/batch_normalization/ReadVariableOpщ
5module_wrapper_2/batch_normalization/ReadVariableOp_1ReadVariableOp>module_wrapper_2_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype027
5module_wrapper_2/batch_normalization/ReadVariableOp_1Ц
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpЬ
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1▌
5module_wrapper_2/batch_normalization/FusedBatchNormV3FusedBatchNormV3-module_wrapper_1/activation/Elu:activations:0;module_wrapper_2/batch_normalization/ReadVariableOp:value:0=module_wrapper_2/batch_normalization/ReadVariableOp_1:value:0Lmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Nmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         dd : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<27
5module_wrapper_2/batch_normalization/FusedBatchNormV3√
3module_wrapper_2/batch_normalization/AssignNewValueAssignVariableOpMmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_resourceBmodule_wrapper_2/batch_normalization/FusedBatchNormV3:batch_mean:0E^module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3module_wrapper_2/batch_normalization/AssignNewValueЗ
5module_wrapper_2/batch_normalization/AssignNewValue_1AssignVariableOpOmodule_wrapper_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceFmodule_wrapper_2/batch_normalization/FusedBatchNormV3:batch_variance:0G^module_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5module_wrapper_2/batch_normalization/AssignNewValue_1Г
&module_wrapper_3/max_pooling2d/MaxPoolMaxPool9module_wrapper_2/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2(
&module_wrapper_3/max_pooling2d/MaxPoolу
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_4_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype021
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOpЪ
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
0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOpЁ
!module_wrapper_4/conv2d_1/BiasAddBiasAdd)module_wrapper_4/conv2d_1/Conv2D:output:08module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !!@2#
!module_wrapper_4/conv2d_1/BiasAdd│
!module_wrapper_5/activation_1/EluElu*module_wrapper_4/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         !!@2#
!module_wrapper_5/activation_1/Eluщ
5module_wrapper_6/batch_normalization_1/ReadVariableOpReadVariableOp>module_wrapper_6_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype027
5module_wrapper_6/batch_normalization_1/ReadVariableOpя
7module_wrapper_6/batch_normalization_1/ReadVariableOp_1ReadVariableOp@module_wrapper_6_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7module_wrapper_6/batch_normalization_1/ReadVariableOp_1Ь
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02H
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpв
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ы
7module_wrapper_6/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3/module_wrapper_5/activation_1/Elu:activations:0=module_wrapper_6/batch_normalization_1/ReadVariableOp:value:0?module_wrapper_6/batch_normalization_1/ReadVariableOp_1:value:0Nmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Pmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !!@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<29
7module_wrapper_6/batch_normalization_1/FusedBatchNormV3Е
5module_wrapper_6/batch_normalization_1/AssignNewValueAssignVariableOpOmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceDmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3:batch_mean:0G^module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype027
5module_wrapper_6/batch_normalization_1/AssignNewValueС
7module_wrapper_6/batch_normalization_1/AssignNewValue_1AssignVariableOpQmodule_wrapper_6_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceHmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3:batch_variance:0I^module_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype029
7module_wrapper_6/batch_normalization_1/AssignNewValue_1Й
(module_wrapper_7/max_pooling2d_1/MaxPoolMaxPool;module_wrapper_6/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2*
(module_wrapper_7/max_pooling2d_1/MaxPoolу
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_8_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype021
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOpЬ
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
0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOpЁ
!module_wrapper_8/conv2d_2/BiasAddBiasAdd)module_wrapper_8/conv2d_2/Conv2D:output:08module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2#
!module_wrapper_8/conv2d_2/BiasAdd│
!module_wrapper_9/activation_2/EluElu*module_wrapper_8/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2#
!module_wrapper_9/activation_2/Eluь
6module_wrapper_10/batch_normalization_2/ReadVariableOpReadVariableOp?module_wrapper_10_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype028
6module_wrapper_10/batch_normalization_2/ReadVariableOpЄ
8module_wrapper_10/batch_normalization_2/ReadVariableOp_1ReadVariableOpAmodule_wrapper_10_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8module_wrapper_10/batch_normalization_2/ReadVariableOp_1Я
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpе
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ё
8module_wrapper_10/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3/module_wrapper_9/activation_2/Elu:activations:0>module_wrapper_10/batch_normalization_2/ReadVariableOp:value:0@module_wrapper_10/batch_normalization_2/ReadVariableOp_1:value:0Omodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Qmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2:
8module_wrapper_10/batch_normalization_2/FusedBatchNormV3К
6module_wrapper_10/batch_normalization_2/AssignNewValueAssignVariableOpPmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceEmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3:batch_mean:0H^module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype028
6module_wrapper_10/batch_normalization_2/AssignNewValueЦ
8module_wrapper_10/batch_normalization_2/AssignNewValue_1AssignVariableOpRmodule_wrapper_10_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceImodule_wrapper_10/batch_normalization_2/FusedBatchNormV3:batch_variance:0J^module_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02:
8module_wrapper_10/batch_normalization_2/AssignNewValue_1М
)module_wrapper_11/max_pooling2d_2/MaxPoolMaxPool<module_wrapper_10/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2+
)module_wrapper_11/max_pooling2d_2/MaxPoolч
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpReadVariableOp9module_wrapper_12_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype022
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOpб
!module_wrapper_12/conv2d_3/Conv2DConv2D2module_wrapper_11/max_pooling2d_2/MaxPool:output:08module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2#
!module_wrapper_12/conv2d_3/Conv2D▐
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp:module_wrapper_12_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype023
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOpї
"module_wrapper_12/conv2d_3/BiasAddBiasAdd*module_wrapper_12/conv2d_3/Conv2D:output:09module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2$
"module_wrapper_12/conv2d_3/BiasAdd╖
"module_wrapper_13/activation_3/EluElu+module_wrapper_12/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:         А2$
"module_wrapper_13/activation_3/Eluэ
6module_wrapper_14/batch_normalization_3/ReadVariableOpReadVariableOp?module_wrapper_14_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6module_wrapper_14/batch_normalization_3/ReadVariableOpє
8module_wrapper_14/batch_normalization_3/ReadVariableOp_1ReadVariableOpAmodule_wrapper_14_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8module_wrapper_14/batch_normalization_3/ReadVariableOp_1а
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02I
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpж
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02K
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ў
8module_wrapper_14/batch_normalization_3/FusedBatchNormV3FusedBatchNormV30module_wrapper_13/activation_3/Elu:activations:0>module_wrapper_14/batch_normalization_3/ReadVariableOp:value:0@module_wrapper_14/batch_normalization_3/ReadVariableOp_1:value:0Omodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Qmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2:
8module_wrapper_14/batch_normalization_3/FusedBatchNormV3К
6module_wrapper_14/batch_normalization_3/AssignNewValueAssignVariableOpPmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceEmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3:batch_mean:0H^module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype028
6module_wrapper_14/batch_normalization_3/AssignNewValueЦ
8module_wrapper_14/batch_normalization_3/AssignNewValue_1AssignVariableOpRmodule_wrapper_14_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceImodule_wrapper_14/batch_normalization_3/FusedBatchNormV3:batch_variance:0J^module_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02:
8module_wrapper_14/batch_normalization_3/AssignNewValue_1Н
)module_wrapper_15/max_pooling2d_3/MaxPoolMaxPool<module_wrapper_14/batch_normalization_3/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2+
)module_wrapper_15/max_pooling2d_3/MaxPoolЧ
'module_wrapper_16/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2)
'module_wrapper_16/dropout/dropout/ConstЎ
%module_wrapper_16/dropout/dropout/MulMul2module_wrapper_15/max_pooling2d_3/MaxPool:output:00module_wrapper_16/dropout/dropout/Const:output:0*
T0*0
_output_shapes
:         А2'
%module_wrapper_16/dropout/dropout/Mul┤
'module_wrapper_16/dropout/dropout/ShapeShape2module_wrapper_15/max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:2)
'module_wrapper_16/dropout/dropout/ShapeЛ
>module_wrapper_16/dropout/dropout/random_uniform/RandomUniformRandomUniform0module_wrapper_16/dropout/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02@
>module_wrapper_16/dropout/dropout/random_uniform/RandomUniformй
0module_wrapper_16/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>22
0module_wrapper_16/dropout/dropout/GreaterEqual/yп
.module_wrapper_16/dropout/dropout/GreaterEqualGreaterEqualGmodule_wrapper_16/dropout/dropout/random_uniform/RandomUniform:output:09module_wrapper_16/dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А20
.module_wrapper_16/dropout/dropout/GreaterEqual╓
&module_wrapper_16/dropout/dropout/CastCast2module_wrapper_16/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2(
&module_wrapper_16/dropout/dropout/Castы
'module_wrapper_16/dropout/dropout/Mul_1Mul)module_wrapper_16/dropout/dropout/Mul:z:0*module_wrapper_16/dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2)
'module_wrapper_16/dropout/dropout/Mul_1У
module_wrapper_17/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
module_wrapper_17/flatten/Const█
!module_wrapper_17/flatten/ReshapeReshape+module_wrapper_16/dropout/dropout/Mul_1:z:0(module_wrapper_17/flatten/Const:output:0*
T0*(
_output_shapes
:         А2#
!module_wrapper_17/flatten/Reshape╫
-module_wrapper_18/dense/MatMul/ReadVariableOpReadVariableOp6module_wrapper_18_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02/
-module_wrapper_18/dense/MatMul/ReadVariableOpр
module_wrapper_18/dense/MatMulMatMul*module_wrapper_17/flatten/Reshape:output:05module_wrapper_18/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
module_wrapper_18/dense/MatMul╒
.module_wrapper_18/dense/BiasAdd/ReadVariableOpReadVariableOp7module_wrapper_18_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.module_wrapper_18/dense/BiasAdd/ReadVariableOpт
module_wrapper_18/dense/BiasAddBiasAdd(module_wrapper_18/dense/MatMul:product:06module_wrapper_18/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
module_wrapper_18/dense/BiasAddм
"module_wrapper_19/activation_4/EluElu(module_wrapper_18/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2$
"module_wrapper_19/activation_4/Elu┌
Fmodule_wrapper_20/batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fmodule_wrapper_20/batch_normalization_4/moments/mean/reduction_indices▓
4module_wrapper_20/batch_normalization_4/moments/meanMean0module_wrapper_19/activation_4/Elu:activations:0Omodule_wrapper_20/batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(26
4module_wrapper_20/batch_normalization_4/moments/meanї
<module_wrapper_20/batch_normalization_4/moments/StopGradientStopGradient=module_wrapper_20/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	А2>
<module_wrapper_20/batch_normalization_4/moments/StopGradient╟
Amodule_wrapper_20/batch_normalization_4/moments/SquaredDifferenceSquaredDifference0module_wrapper_19/activation_4/Elu:activations:0Emodule_wrapper_20/batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:         А2C
Amodule_wrapper_20/batch_normalization_4/moments/SquaredDifferenceт
Jmodule_wrapper_20/batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2L
Jmodule_wrapper_20/batch_normalization_4/moments/variance/reduction_indices╙
8module_wrapper_20/batch_normalization_4/moments/varianceMeanEmodule_wrapper_20/batch_normalization_4/moments/SquaredDifference:z:0Smodule_wrapper_20/batch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2:
8module_wrapper_20/batch_normalization_4/moments/variance∙
7module_wrapper_20/batch_normalization_4/moments/SqueezeSqueeze=module_wrapper_20/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 29
7module_wrapper_20/batch_normalization_4/moments/SqueezeБ
9module_wrapper_20/batch_normalization_4/moments/Squeeze_1SqueezeAmodule_wrapper_20/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2;
9module_wrapper_20/batch_normalization_4/moments/Squeeze_1├
=module_wrapper_20/batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2?
=module_wrapper_20/batch_normalization_4/AssignMovingAvg/decayЭ
Fmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOpOmodule_wrapper_20_batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02H
Fmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOp╣
;module_wrapper_20/batch_normalization_4/AssignMovingAvg/subSubNmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0@module_wrapper_20/batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes	
:А2=
;module_wrapper_20/batch_normalization_4/AssignMovingAvg/sub░
;module_wrapper_20/batch_normalization_4/AssignMovingAvg/mulMul?module_wrapper_20/batch_normalization_4/AssignMovingAvg/sub:z:0Fmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2=
;module_wrapper_20/batch_normalization_4/AssignMovingAvg/mulЗ
7module_wrapper_20/batch_normalization_4/AssignMovingAvgAssignSubVariableOpOmodule_wrapper_20_batch_normalization_4_assignmovingavg_readvariableop_resource?module_wrapper_20/batch_normalization_4/AssignMovingAvg/mul:z:0G^module_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype029
7module_wrapper_20/batch_normalization_4/AssignMovingAvg╟
?module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2A
?module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/decayг
Hmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOpQmodule_wrapper_20_batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02J
Hmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp┴
=module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/subSubPmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:0Bmodule_wrapper_20/batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2?
=module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/sub╕
=module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/mulMulAmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/sub:z:0Hmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2?
=module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/mulС
9module_wrapper_20/batch_normalization_4/AssignMovingAvg_1AssignSubVariableOpQmodule_wrapper_20_batch_normalization_4_assignmovingavg_1_readvariableop_resourceAmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/mul:z:0I^module_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02;
9module_wrapper_20/batch_normalization_4/AssignMovingAvg_1№
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpReadVariableOpDmodule_wrapper_20_batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOpВ
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOpReadVariableOpFmodule_wrapper_20_batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:А*
dtype02?
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp╖
7module_wrapper_20/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7module_wrapper_20/batch_normalization_4/batchnorm/add/yг
5module_wrapper_20/batch_normalization_4/batchnorm/addAddV2Bmodule_wrapper_20/batch_normalization_4/moments/Squeeze_1:output:0@module_wrapper_20/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А27
5module_wrapper_20/batch_normalization_4/batchnorm/add▄
7module_wrapper_20/batch_normalization_4/batchnorm/RsqrtRsqrt9module_wrapper_20/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А29
7module_wrapper_20/batch_normalization_4/batchnorm/RsqrtЯ
5module_wrapper_20/batch_normalization_4/batchnorm/mulMul;module_wrapper_20/batch_normalization_4/batchnorm/Rsqrt:y:0Emodule_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:А27
5module_wrapper_20/batch_normalization_4/batchnorm/mulЩ
7module_wrapper_20/batch_normalization_4/batchnorm/mul_1Mul0module_wrapper_19/activation_4/Elu:activations:09module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А29
7module_wrapper_20/batch_normalization_4/batchnorm/mul_1Ь
7module_wrapper_20/batch_normalization_4/batchnorm/mul_2Mul@module_wrapper_20/batch_normalization_4/moments/Squeeze:output:09module_wrapper_20/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А29
7module_wrapper_20/batch_normalization_4/batchnorm/mul_2Э
5module_wrapper_20/batch_normalization_4/batchnorm/subSubCmodule_wrapper_20/batch_normalization_4/Cast/ReadVariableOp:value:0;module_wrapper_20/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А27
5module_wrapper_20/batch_normalization_4/batchnorm/subж
7module_wrapper_20/batch_normalization_4/batchnorm/add_1AddV2;module_wrapper_20/batch_normalization_4/batchnorm/mul_1:z:09module_wrapper_20/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А29
7module_wrapper_20/batch_normalization_4/batchnorm/add_1Ы
)module_wrapper_21/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)module_wrapper_21/dropout_1/dropout/Const¤
'module_wrapper_21/dropout_1/dropout/MulMul;module_wrapper_20/batch_normalization_4/batchnorm/add_1:z:02module_wrapper_21/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:         А2)
'module_wrapper_21/dropout_1/dropout/Mul┴
)module_wrapper_21/dropout_1/dropout/ShapeShape;module_wrapper_20/batch_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2+
)module_wrapper_21/dropout_1/dropout/ShapeЙ
@module_wrapper_21/dropout_1/dropout/random_uniform/RandomUniformRandomUniform2module_wrapper_21/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02B
@module_wrapper_21/dropout_1/dropout/random_uniform/RandomUniformн
2module_wrapper_21/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2module_wrapper_21/dropout_1/dropout/GreaterEqual/yп
0module_wrapper_21/dropout_1/dropout/GreaterEqualGreaterEqualImodule_wrapper_21/dropout_1/dropout/random_uniform/RandomUniform:output:0;module_wrapper_21/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А22
0module_wrapper_21/dropout_1/dropout/GreaterEqual╘
(module_wrapper_21/dropout_1/dropout/CastCast4module_wrapper_21/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2*
(module_wrapper_21/dropout_1/dropout/Castы
)module_wrapper_21/dropout_1/dropout/Mul_1Mul+module_wrapper_21/dropout_1/dropout/Mul:z:0,module_wrapper_21/dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2+
)module_wrapper_21/dropout_1/dropout/Mul_1▄
/module_wrapper_22/dense_1/MatMul/ReadVariableOpReadVariableOp8module_wrapper_22_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype021
/module_wrapper_22/dense_1/MatMul/ReadVariableOpш
 module_wrapper_22/dense_1/MatMulMatMul-module_wrapper_21/dropout_1/dropout/Mul_1:z:07module_wrapper_22/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 module_wrapper_22/dense_1/MatMul┌
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_22_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOpщ
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
8module_wrapper_10/batch_normalization_2/AssignNewValue_18module_wrapper_10/batch_normalization_2/AssignNewValue_12Т
Gmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOpGmodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2Ц
Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Imodule_wrapper_10/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12p
6module_wrapper_10/batch_normalization_2/ReadVariableOp6module_wrapper_10/batch_normalization_2/ReadVariableOp2t
8module_wrapper_10/batch_normalization_2/ReadVariableOp_18module_wrapper_10/batch_normalization_2/ReadVariableOp_12f
1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp1module_wrapper_12/conv2d_3/BiasAdd/ReadVariableOp2d
0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp0module_wrapper_12/conv2d_3/Conv2D/ReadVariableOp2p
6module_wrapper_14/batch_normalization_3/AssignNewValue6module_wrapper_14/batch_normalization_3/AssignNewValue2t
8module_wrapper_14/batch_normalization_3/AssignNewValue_18module_wrapper_14/batch_normalization_3/AssignNewValue_12Т
Gmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOpGmodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2Ц
Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Imodule_wrapper_14/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12p
6module_wrapper_14/batch_normalization_3/ReadVariableOp6module_wrapper_14/batch_normalization_3/ReadVariableOp2t
8module_wrapper_14/batch_normalization_3/ReadVariableOp_18module_wrapper_14/batch_normalization_3/ReadVariableOp_12`
.module_wrapper_18/dense/BiasAdd/ReadVariableOp.module_wrapper_18/dense/BiasAdd/ReadVariableOp2^
-module_wrapper_18/dense/MatMul/ReadVariableOp-module_wrapper_18/dense/MatMul/ReadVariableOp2j
3module_wrapper_2/batch_normalization/AssignNewValue3module_wrapper_2/batch_normalization/AssignNewValue2n
5module_wrapper_2/batch_normalization/AssignNewValue_15module_wrapper_2/batch_normalization/AssignNewValue_12М
Dmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOpDmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp2Р
Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Fmodule_wrapper_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3module_wrapper_2/batch_normalization/ReadVariableOp3module_wrapper_2/batch_normalization/ReadVariableOp2n
5module_wrapper_2/batch_normalization/ReadVariableOp_15module_wrapper_2/batch_normalization/ReadVariableOp_12r
7module_wrapper_20/batch_normalization_4/AssignMovingAvg7module_wrapper_20/batch_normalization_4/AssignMovingAvg2Р
Fmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOpFmodule_wrapper_20/batch_normalization_4/AssignMovingAvg/ReadVariableOp2v
9module_wrapper_20/batch_normalization_4/AssignMovingAvg_19module_wrapper_20/batch_normalization_4/AssignMovingAvg_12Ф
Hmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpHmodule_wrapper_20/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2z
;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp;module_wrapper_20/batch_normalization_4/Cast/ReadVariableOp2~
=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp=module_wrapper_20/batch_normalization_4/Cast_1/ReadVariableOp2d
0module_wrapper_22/dense_1/BiasAdd/ReadVariableOp0module_wrapper_22/dense_1/BiasAdd/ReadVariableOp2b
/module_wrapper_22/dense_1/MatMul/ReadVariableOp/module_wrapper_22/dense_1/MatMul/ReadVariableOp2d
0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp0module_wrapper_4/conv2d_1/BiasAdd/ReadVariableOp2b
/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp/module_wrapper_4/conv2d_1/Conv2D/ReadVariableOp2n
5module_wrapper_6/batch_normalization_1/AssignNewValue5module_wrapper_6/batch_normalization_1/AssignNewValue2r
7module_wrapper_6/batch_normalization_1/AssignNewValue_17module_wrapper_6/batch_normalization_1/AssignNewValue_12Р
Fmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOpFmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2Ф
Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Hmodule_wrapper_6/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5module_wrapper_6/batch_normalization_1/ReadVariableOp5module_wrapper_6/batch_normalization_1/ReadVariableOp2r
7module_wrapper_6/batch_normalization_1/ReadVariableOp_17module_wrapper_6/batch_normalization_1/ReadVariableOp_12d
0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp0module_wrapper_8/conv2d_2/BiasAdd/ReadVariableOp2b
/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp/module_wrapper_8/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
ь
M
1__inference_module_wrapper_11_layer_call_fn_97783

args_0
identity╥
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
GPU 2J 8В *U
fPRN
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_953782
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
Х
╦
0__inference_module_wrapper_2_layer_call_fn_97444

args_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЭ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_946882
StatefulPartitionedCallЦ
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
С
h
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_98135

args_0
identityo
dropout_1/IdentityIdentityargs_0*
T0*(
_output_shapes
:         А2
dropout_1/Identityp
IdentityIdentitydropout_1/Identity:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameargs_0
╤
е
0__inference_module_wrapper_8_layer_call_fn_97671

args_0!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallГ
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
GPU 2J 8В *T
fORM
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_954642
StatefulPartitionedCallЦ
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
 
_user_specified_nameargs_0"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╓
serving_default┬
]
module_wrapper_inputE
&serving_default_module_wrapper_input:0         ddE
module_wrapper_230
StatefulPartitionedCall:0         tensorflow/serving/predict:╤╥
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
╢_default_save_signature
╖__call__
+╕&call_and_return_all_conditional_losses"Щ
_tf_keras_sequential·{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "module_wrapper_input"}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}]}, "shared_object_id": 1, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "float32", "module_wrapper_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 2}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
╗
_module
 trainable_variables
!regularization_losses
"	variables
#	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"name": "module_wrapper", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
$_module
%trainable_variables
&regularization_losses
'	variables
(	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"name": "module_wrapper_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
)_module
*trainable_variables
+regularization_losses
,	variables
-	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"name": "module_wrapper_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
._module
/trainable_variables
0regularization_losses
1	variables
2	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"name": "module_wrapper_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
3_module
4trainable_variables
5regularization_losses
6	variables
7	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"name": "module_wrapper_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
8_module
9trainable_variables
:regularization_losses
;	variables
<	keras_api
├__call__
+─&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"name": "module_wrapper_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
=_module
>trainable_variables
?regularization_losses
@	variables
A	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"name": "module_wrapper_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
B_module
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"name": "module_wrapper_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
G_module
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"name": "module_wrapper_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
L_module
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"name": "module_wrapper_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
Q_module
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
═__call__
+╬&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
V_module
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
╧__call__
+╨&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
[_module
\trainable_variables
]regularization_losses
^	variables
_	keras_api
╤__call__
+╥&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
`_module
atrainable_variables
bregularization_losses
c	variables
d	keras_api
╙__call__
+╘&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
e_module
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
╒__call__
+╓&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
j_module
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
╫__call__
+╪&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
o_module
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
t_module
utrainable_variables
vregularization_losses
w	variables
x	keras_api
█__call__
+▄&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
y_module
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
▌__call__
+▐&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
┴
~_module
trainable_variables
Аregularization_losses
Б	variables
В	keras_api
▀__call__
+р&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
├
Г_module
Дtrainable_variables
Еregularization_losses
Ж	variables
З	keras_api
с__call__
+т&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
├
И_module
Йtrainable_variables
Кregularization_losses
Л	variables
М	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
├
Н_module
Оtrainable_variables
Пregularization_losses
Р	variables
С	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
├
Т_module
Уtrainable_variables
Фregularization_losses
Х	variables
Ц	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "module_wrapper_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
┘
	Чiter

Шdecay
Щlearning_rate
ЪmomentumЫmomentumаЬmomentumбЭmomentumвЮmomentumгЯmomentumдаmomentumебmomentumжвmomentumзгmomentumидmomentumйеmomentumкжmomentumлзmomentumмиmomentumнйmomentumокmomentumплmomentum░мmomentum▒нmomentum▓оmomentum│пmomentum┤░momentum╡"
	optimizer
▄
Ы0
Ь1
Э2
Ю3
Я4
а5
б6
в7
г8
д9
е10
ж11
з12
и13
й14
к15
л16
м17
н18
о19
п20
░21"
trackable_list_wrapper
 "
trackable_list_wrapper
╢
Ы0
Ь1
Э2
Ю3
▒4
▓5
Я6
а7
б8
в9
│10
┤11
г12
д13
е14
ж15
╡16
╢17
з18
и19
й20
к21
╖22
╕23
л24
м25
н26
о27
╣28
║29
п30
░31"
trackable_list_wrapper
╙
╗metrics
trainable_variables
regularization_losses
 ╝layer_regularization_losses
	variables
╜non_trainable_variables
╛layer_metrics
┐layers
╖__call__
╢_default_save_signature
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
-
щserving_default"
signature_map
¤

Ыkernel
	Ьbias
└trainable_variables
┴regularization_losses
┬	variables
├	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses"╨	
_tf_keras_layer╢	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 3]}}
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
╡
─metrics
 trainable_variables
!regularization_losses
 ┼layer_regularization_losses
"	variables
╞non_trainable_variables
╟layer_metrics
╚layers
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
╓
╔trainable_variables
╩regularization_losses
╦	variables
╠	keras_api
ь__call__
+э&call_and_return_all_conditional_losses"┴
_tf_keras_layerз{"name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "elu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
═metrics
%trainable_variables
&regularization_losses
 ╬layer_regularization_losses
'	variables
╧non_trainable_variables
╨layer_metrics
╤layers
╗__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
├	
	╥axis

Эgamma
	Юbeta
▒moving_mean
▓moving_variance
╙trainable_variables
╘regularization_losses
╒	variables
╓	keras_api
ю__call__
+я&call_and_return_all_conditional_losses"ф
_tf_keras_layer╩{"name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 32]}}
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Э0
Ю1
▒2
▓3"
trackable_list_wrapper
╡
╫metrics
*trainable_variables
+regularization_losses
 ╪layer_regularization_losses
,	variables
┘non_trainable_variables
┌layer_metrics
█layers
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
Б
▄trainable_variables
▌regularization_losses
▐	variables
▀	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses"ь
_tf_keras_layer╥{"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
рmetrics
/trainable_variables
0regularization_losses
 сlayer_regularization_losses
1	variables
тnon_trainable_variables
уlayer_metrics
фlayers
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
№	
Яkernel
	аbias
хtrainable_variables
цregularization_losses
ч	variables
ш	keras_api
Є__call__
+є&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 32]}}
0
Я0
а1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Я0
а1"
trackable_list_wrapper
╡
щmetrics
4trainable_variables
5regularization_losses
 ъlayer_regularization_losses
6	variables
ыnon_trainable_variables
ьlayer_metrics
эlayers
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
┌
юtrainable_variables
яregularization_losses
Ё	variables
ё	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses"┼
_tf_keras_layerл{"name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "elu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Єmetrics
9trainable_variables
:regularization_losses
 єlayer_regularization_losses
;	variables
Їnon_trainable_variables
їlayer_metrics
Ўlayers
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
┼	
	ўaxis

бgamma
	вbeta
│moving_mean
┤moving_variance
°trainable_variables
∙regularization_losses
·	variables
√	keras_api
Ў__call__
+ў&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 64]}}
0
б0
в1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
б0
в1
│2
┤3"
trackable_list_wrapper
╡
№metrics
>trainable_variables
?regularization_losses
 ¤layer_regularization_losses
@	variables
■non_trainable_variables
 layer_metrics
Аlayers
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
Е
Бtrainable_variables
Вregularization_losses
Г	variables
Д	keras_api
°__call__
+∙&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Еmetrics
Ctrainable_variables
Dregularization_losses
 Жlayer_regularization_losses
E	variables
Зnon_trainable_variables
Иlayer_metrics
Йlayers
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
№	
гkernel
	дbias
Кtrainable_variables
Лregularization_losses
М	variables
Н	keras_api
·__call__
+√&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
0
г0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
г0
д1"
trackable_list_wrapper
╡
Оmetrics
Htrainable_variables
Iregularization_losses
 Пlayer_regularization_losses
J	variables
Рnon_trainable_variables
Сlayer_metrics
Тlayers
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
┌
Уtrainable_variables
Фregularization_losses
Х	variables
Ц	keras_api
№__call__
+¤&call_and_return_all_conditional_losses"┼
_tf_keras_layerл{"name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "elu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Чmetrics
Mtrainable_variables
Nregularization_losses
 Шlayer_regularization_losses
O	variables
Щnon_trainable_variables
Ъlayer_metrics
Ыlayers
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
┼	
	Ьaxis

еgamma
	жbeta
╡moving_mean
╢moving_variance
Эtrainable_variables
Юregularization_losses
Я	variables
а	keras_api
■__call__
+ &call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
0
е0
ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
е0
ж1
╡2
╢3"
trackable_list_wrapper
╡
бmetrics
Rtrainable_variables
Sregularization_losses
 вlayer_regularization_losses
T	variables
гnon_trainable_variables
дlayer_metrics
еlayers
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
Е
жtrainable_variables
зregularization_losses
и	variables
й	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
кmetrics
Wtrainable_variables
Xregularization_losses
 лlayer_regularization_losses
Y	variables
мnon_trainable_variables
нlayer_metrics
оlayers
╧__call__
+╨&call_and_return_all_conditional_losses
'╨"call_and_return_conditional_losses"
_generic_user_object
√	
зkernel
	иbias
пtrainable_variables
░regularization_losses
▒	variables
▓	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
0
з0
и1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
з0
и1"
trackable_list_wrapper
╡
│metrics
\trainable_variables
]regularization_losses
 ┤layer_regularization_losses
^	variables
╡non_trainable_variables
╢layer_metrics
╖layers
╤__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses"
_generic_user_object
┌
╕trainable_variables
╣regularization_losses
║	variables
╗	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"┼
_tf_keras_layerл{"name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "elu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╝metrics
atrainable_variables
bregularization_losses
 ╜layer_regularization_losses
c	variables
╛non_trainable_variables
┐layer_metrics
└layers
╙__call__
+╘&call_and_return_all_conditional_losses
'╘"call_and_return_conditional_losses"
_generic_user_object
┼	
	┴axis

йgamma
	кbeta
╖moving_mean
╕moving_variance
┬trainable_variables
├regularization_losses
─	variables
┼	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
0
й0
к1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
й0
к1
╖2
╕3"
trackable_list_wrapper
╡
╞metrics
ftrainable_variables
gregularization_losses
 ╟layer_regularization_losses
h	variables
╚non_trainable_variables
╔layer_metrics
╩layers
╒__call__
+╓&call_and_return_all_conditional_losses
'╓"call_and_return_conditional_losses"
_generic_user_object
Е
╦trainable_variables
╠regularization_losses
═	variables
╬	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╧metrics
ktrainable_variables
lregularization_losses
 ╨layer_regularization_losses
m	variables
╤non_trainable_variables
╥layer_metrics
╙layers
╫__call__
+╪&call_and_return_all_conditional_losses
'╪"call_and_return_conditional_losses"
_generic_user_object
ч
╘trainable_variables
╒regularization_losses
╓	variables
╫	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"╥
_tf_keras_layer╕{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╪metrics
ptrainable_variables
qregularization_losses
 ┘layer_regularization_losses
r	variables
┌non_trainable_variables
█layer_metrics
▄layers
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
ш
▌trainable_variables
▐regularization_losses
▀	variables
р	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"╙
_tf_keras_layer╣{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
сmetrics
utrainable_variables
vregularization_losses
 тlayer_regularization_losses
w	variables
уnon_trainable_variables
фlayer_metrics
хlayers
█__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses"
_generic_user_object
√
лkernel
	мbias
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
О__call__
+П&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
0
л0
м1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
л0
м1"
trackable_list_wrapper
╡
ъmetrics
ztrainable_variables
{regularization_losses
 ыlayer_regularization_losses
|	variables
ьnon_trainable_variables
эlayer_metrics
юlayers
▌__call__
+▐&call_and_return_all_conditional_losses
'▐"call_and_return_conditional_losses"
_generic_user_object
┌
яtrainable_variables
Ёregularization_losses
ё	variables
Є	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"┼
_tf_keras_layerл{"name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "elu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╖
єmetrics
trainable_variables
Аregularization_losses
 Їlayer_regularization_losses
Б	variables
їnon_trainable_variables
Ўlayer_metrics
ўlayers
▀__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
┐	
	°axis

нgamma
	оbeta
╣moving_mean
║moving_variance
∙trainable_variables
·regularization_losses
√	variables
№	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"р
_tf_keras_layer╞{"name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
0
н0
о1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
н0
о1
╣2
║3"
trackable_list_wrapper
╕
¤metrics
Дtrainable_variables
Еregularization_losses
 ■layer_regularization_losses
Ж	variables
 non_trainable_variables
Аlayer_metrics
Бlayers
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
ы
Вtrainable_variables
Гregularization_losses
Д	variables
Е	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"╓
_tf_keras_layer╝{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Жmetrics
Йtrainable_variables
Кregularization_losses
 Зlayer_regularization_losses
Л	variables
Иnon_trainable_variables
Йlayer_metrics
Кlayers
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
№
пkernel
	░bias
Лtrainable_variables
Мregularization_losses
Н	variables
О	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 29, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
0
п0
░1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
п0
░1"
trackable_list_wrapper
╕
Пmetrics
Оtrainable_variables
Пregularization_losses
 Рlayer_regularization_losses
Р	variables
Сnon_trainable_variables
Тlayer_metrics
Уlayers
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
▐
Фtrainable_variables
Хregularization_losses
Ц	variables
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"╔
_tf_keras_layerп{"name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "softmax"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Шmetrics
Уtrainable_variables
Фregularization_losses
 Щlayer_regularization_losses
Х	variables
Ъnon_trainable_variables
Ыlayer_metrics
Ьlayers
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
6:4 2module_wrapper/conv2d/kernel
(:& 2module_wrapper/conv2d/bias
8:6 2*module_wrapper_2/batch_normalization/gamma
7:5 2)module_wrapper_2/batch_normalization/beta
::8 @2 module_wrapper_4/conv2d_1/kernel
,:*@2module_wrapper_4/conv2d_1/bias
::8@2,module_wrapper_6/batch_normalization_1/gamma
9:7@2+module_wrapper_6/batch_normalization_1/beta
::8@@2 module_wrapper_8/conv2d_2/kernel
,:*@2module_wrapper_8/conv2d_2/bias
;:9@2-module_wrapper_10/batch_normalization_2/gamma
::8@2,module_wrapper_10/batch_normalization_2/beta
<::@А2!module_wrapper_12/conv2d_3/kernel
.:,А2module_wrapper_12/conv2d_3/bias
<::А2-module_wrapper_14/batch_normalization_3/gamma
;:9А2,module_wrapper_14/batch_normalization_3/beta
2:0
АА2module_wrapper_18/dense/kernel
+:)А2module_wrapper_18/dense/bias
<::А2-module_wrapper_20/batch_normalization_4/gamma
;:9А2,module_wrapper_20/batch_normalization_4/beta
3:1	А2 module_wrapper_22/dense_1/kernel
,:*2module_wrapper_22/dense_1/bias
@:>  (20module_wrapper_2/batch_normalization/moving_mean
D:B  (24module_wrapper_2/batch_normalization/moving_variance
B:@@ (22module_wrapper_6/batch_normalization_1/moving_mean
F:D@ (26module_wrapper_6/batch_normalization_1/moving_variance
C:A@ (23module_wrapper_10/batch_normalization_2/moving_mean
G:E@ (27module_wrapper_10/batch_normalization_2/moving_variance
D:BА (23module_wrapper_14/batch_normalization_3/moving_mean
H:FА (27module_wrapper_14/batch_normalization_3/moving_variance
D:BА (23module_wrapper_20/batch_normalization_4/moving_mean
H:FА (27module_wrapper_20/batch_normalization_4/moving_variance
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
p
▒0
▓1
│2
┤3
╡4
╢5
╖6
╕7
╣8
║9"
trackable_list_wrapper
 "
trackable_dict_wrapper
╓
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
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
╕
Яmetrics
└trainable_variables
┴regularization_losses
 аlayer_regularization_losses
┬	variables
бnon_trainable_variables
вlayer_metrics
гlayers
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
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
╕
дmetrics
╔trainable_variables
╩regularization_losses
 еlayer_regularization_losses
╦	variables
жnon_trainable_variables
зlayer_metrics
иlayers
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
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
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Э0
Ю1
▒2
▓3"
trackable_list_wrapper
╕
йmetrics
╙trainable_variables
╘regularization_losses
 кlayer_regularization_losses
╒	variables
лnon_trainable_variables
мlayer_metrics
нlayers
ю__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
▒0
▓1"
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
╕
оmetrics
▄trainable_variables
▌regularization_losses
 пlayer_regularization_losses
▐	variables
░non_trainable_variables
▒layer_metrics
▓layers
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
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
Я0
а1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Я0
а1"
trackable_list_wrapper
╕
│metrics
хtrainable_variables
цregularization_losses
 ┤layer_regularization_losses
ч	variables
╡non_trainable_variables
╢layer_metrics
╖layers
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
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
╕
╕metrics
юtrainable_variables
яregularization_losses
 ╣layer_regularization_losses
Ё	variables
║non_trainable_variables
╗layer_metrics
╝layers
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
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
0
б0
в1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
б0
в1
│2
┤3"
trackable_list_wrapper
╕
╜metrics
°trainable_variables
∙regularization_losses
 ╛layer_regularization_losses
·	variables
┐non_trainable_variables
└layer_metrics
┴layers
Ў__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
│0
┤1"
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
╕
┬metrics
Бtrainable_variables
Вregularization_losses
 ├layer_regularization_losses
Г	variables
─non_trainable_variables
┼layer_metrics
╞layers
°__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses"
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
г0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
г0
д1"
trackable_list_wrapper
╕
╟metrics
Кtrainable_variables
Лregularization_losses
 ╚layer_regularization_losses
М	variables
╔non_trainable_variables
╩layer_metrics
╦layers
·__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
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
╕
╠metrics
Уtrainable_variables
Фregularization_losses
 ═layer_regularization_losses
Х	variables
╬non_trainable_variables
╧layer_metrics
╨layers
№__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
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
0
е0
ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
е0
ж1
╡2
╢3"
trackable_list_wrapper
╕
╤metrics
Эtrainable_variables
Юregularization_losses
 ╥layer_regularization_losses
Я	variables
╙non_trainable_variables
╘layer_metrics
╒layers
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
╡0
╢1"
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
╕
╓metrics
жtrainable_variables
зregularization_losses
 ╫layer_regularization_losses
и	variables
╪non_trainable_variables
┘layer_metrics
┌layers
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
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
з0
и1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
з0
и1"
trackable_list_wrapper
╕
█metrics
пtrainable_variables
░regularization_losses
 ▄layer_regularization_losses
▒	variables
▌non_trainable_variables
▐layer_metrics
▀layers
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
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
╕
рmetrics
╕trainable_variables
╣regularization_losses
 сlayer_regularization_losses
║	variables
тnon_trainable_variables
уlayer_metrics
фlayers
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
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
0
й0
к1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
й0
к1
╖2
╕3"
trackable_list_wrapper
╕
хmetrics
┬trainable_variables
├regularization_losses
 цlayer_regularization_losses
─	variables
чnon_trainable_variables
шlayer_metrics
щlayers
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
╖0
╕1"
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
╕
ъmetrics
╦trainable_variables
╠regularization_losses
 ыlayer_regularization_losses
═	variables
ьnon_trainable_variables
эlayer_metrics
юlayers
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
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
╕
яmetrics
╘trainable_variables
╒regularization_losses
 Ёlayer_regularization_losses
╓	variables
ёnon_trainable_variables
Єlayer_metrics
єlayers
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
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
╕
Їmetrics
▌trainable_variables
▐regularization_losses
 їlayer_regularization_losses
▀	variables
Ўnon_trainable_variables
ўlayer_metrics
°layers
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
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
л0
м1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
л0
м1"
trackable_list_wrapper
╕
∙metrics
цtrainable_variables
чregularization_losses
 ·layer_regularization_losses
ш	variables
√non_trainable_variables
№layer_metrics
¤layers
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
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
╕
■metrics
яtrainable_variables
Ёregularization_losses
  layer_regularization_losses
ё	variables
Аnon_trainable_variables
Бlayer_metrics
Вlayers
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
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
0
н0
о1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
н0
о1
╣2
║3"
trackable_list_wrapper
╕
Гmetrics
∙trainable_variables
·regularization_losses
 Дlayer_regularization_losses
√	variables
Еnon_trainable_variables
Жlayer_metrics
Зlayers
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
╣0
║1"
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
╕
Иmetrics
Вtrainable_variables
Гregularization_losses
 Йlayer_regularization_losses
Д	variables
Кnon_trainable_variables
Лlayer_metrics
Мlayers
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
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
п0
░1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
п0
░1"
trackable_list_wrapper
╕
Нmetrics
Лtrainable_variables
Мregularization_losses
 Оlayer_regularization_losses
Н	variables
Пnon_trainable_variables
Рlayer_metrics
Сlayers
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
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
╕
Тmetrics
Фtrainable_variables
Хregularization_losses
 Уlayer_regularization_losses
Ц	variables
Фnon_trainable_variables
Хlayer_metrics
Цlayers
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
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
╫

Чtotal

Шcount
Щ	variables
Ъ	keras_api"Ь
_tf_keras_metricБ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 3}
Ы

Ыtotal

Ьcount
Э
_fn_kwargs
Ю	variables
Я	keras_api"╧
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
▒0
▓1"
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
0
│0
┤1"
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
0
╡0
╢1"
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
0
╖0
╕1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
╣0
║1"
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
Ч0
Ш1"
trackable_list_wrapper
.
Щ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
.
Ю	variables"
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
G:E@А2.SGD/module_wrapper_12/conv2d_3/kernel/momentum
9:7А2,SGD/module_wrapper_12/conv2d_3/bias/momentum
G:EА2:SGD/module_wrapper_14/batch_normalization_3/gamma/momentum
F:DА29SGD/module_wrapper_14/batch_normalization_3/beta/momentum
=:;
АА2+SGD/module_wrapper_18/dense/kernel/momentum
6:4А2)SGD/module_wrapper_18/dense/bias/momentum
G:EА2:SGD/module_wrapper_20/batch_normalization_4/gamma/momentum
F:DА29SGD/module_wrapper_20/batch_normalization_4/beta/momentum
>:<	А2-SGD/module_wrapper_22/dense_1/kernel/momentum
7:52+SGD/module_wrapper_22/dense_1/bias/momentum
є2Ё
 __inference__wrapped_model_94640╦
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *;в8
6К3
module_wrapper_input         dd
Ў2є
*__inference_sequential_layer_call_fn_95045
*__inference_sequential_layer_call_fn_97024
*__inference_sequential_layer_call_fn_97093
*__inference_sequential_layer_call_fn_95982└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
E__inference_sequential_layer_call_and_return_conditional_losses_97219
E__inference_sequential_layer_call_and_return_conditional_losses_97373
E__inference_sequential_layer_call_and_return_conditional_losses_96074
E__inference_sequential_layer_call_and_return_conditional_losses_96166└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ж2г
.__inference_module_wrapper_layer_call_fn_97382
.__inference_module_wrapper_layer_call_fn_97391└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
▄2┘
I__inference_module_wrapper_layer_call_and_return_conditional_losses_97401
I__inference_module_wrapper_layer_call_and_return_conditional_losses_97411└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
0__inference_module_wrapper_1_layer_call_fn_97416
0__inference_module_wrapper_1_layer_call_fn_97421└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
р2▌
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_97426
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_97431└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
0__inference_module_wrapper_2_layer_call_fn_97444
0__inference_module_wrapper_2_layer_call_fn_97457└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
р2▌
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_97475
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_97493└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
0__inference_module_wrapper_3_layer_call_fn_97498
0__inference_module_wrapper_3_layer_call_fn_97503└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
р2▌
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_97508
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_97513└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
0__inference_module_wrapper_4_layer_call_fn_97522
0__inference_module_wrapper_4_layer_call_fn_97531└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
р2▌
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_97541
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_97551└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
0__inference_module_wrapper_5_layer_call_fn_97556
0__inference_module_wrapper_5_layer_call_fn_97561└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
р2▌
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_97566
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_97571└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
0__inference_module_wrapper_6_layer_call_fn_97584
0__inference_module_wrapper_6_layer_call_fn_97597└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
р2▌
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_97615
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_97633└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
0__inference_module_wrapper_7_layer_call_fn_97638
0__inference_module_wrapper_7_layer_call_fn_97643└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
р2▌
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_97648
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_97653└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
0__inference_module_wrapper_8_layer_call_fn_97662
0__inference_module_wrapper_8_layer_call_fn_97671└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
р2▌
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_97681
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_97691└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
0__inference_module_wrapper_9_layer_call_fn_97696
0__inference_module_wrapper_9_layer_call_fn_97701└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
р2▌
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_97706
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_97711└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_10_layer_call_fn_97724
1__inference_module_wrapper_10_layer_call_fn_97737└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_97755
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_97773└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_11_layer_call_fn_97778
1__inference_module_wrapper_11_layer_call_fn_97783└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_97788
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_97793└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_12_layer_call_fn_97802
1__inference_module_wrapper_12_layer_call_fn_97811└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_97821
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_97831└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_13_layer_call_fn_97836
1__inference_module_wrapper_13_layer_call_fn_97841└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_97846
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_97851└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_14_layer_call_fn_97864
1__inference_module_wrapper_14_layer_call_fn_97877└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_97895
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_97913└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_15_layer_call_fn_97918
1__inference_module_wrapper_15_layer_call_fn_97923└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_97928
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_97933└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_16_layer_call_fn_97938
1__inference_module_wrapper_16_layer_call_fn_97943└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_97948
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_97960└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_17_layer_call_fn_97965
1__inference_module_wrapper_17_layer_call_fn_97970└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_97976
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_97982└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_18_layer_call_fn_97991
1__inference_module_wrapper_18_layer_call_fn_98000└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_98010
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_98020└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_19_layer_call_fn_98025
1__inference_module_wrapper_19_layer_call_fn_98030└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_98035
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_98040└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_20_layer_call_fn_98053
1__inference_module_wrapper_20_layer_call_fn_98066└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_98086
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_98120└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_21_layer_call_fn_98125
1__inference_module_wrapper_21_layer_call_fn_98130└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_98135
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_98147└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_22_layer_call_fn_98156
1__inference_module_wrapper_22_layer_call_fn_98165└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_98175
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_98185└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_23_layer_call_fn_98190
1__inference_module_wrapper_23_layer_call_fn_98195└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_98200
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_98205└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╫B╘
#__inference_signature_wrapper_96241module_wrapper_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
д2б
3__inference_batch_normalization_layer_call_fn_98218
3__inference_batch_normalization_layer_call_fn_98231┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┌2╫
N__inference_batch_normalization_layer_call_and_return_conditional_losses_98249
N__inference_batch_normalization_layer_call_and_return_conditional_losses_98267┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Х2Т
-__inference_max_pooling2d_layer_call_fn_96379р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
░2н
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_96373р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2е
5__inference_batch_normalization_1_layer_call_fn_98280
5__inference_batch_normalization_1_layer_call_fn_98293┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_98311
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_98329┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ч2Ф
/__inference_max_pooling2d_1_layer_call_fn_96517р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▓2п
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_96511р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2е
5__inference_batch_normalization_2_layer_call_fn_98342
5__inference_batch_normalization_2_layer_call_fn_98355┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_98373
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_98391┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ч2Ф
/__inference_max_pooling2d_2_layer_call_fn_96655р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▓2п
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_96649р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2е
5__inference_batch_normalization_3_layer_call_fn_98404
5__inference_batch_normalization_3_layer_call_fn_98417┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_98435
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_98453┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ч2Ф
/__inference_max_pooling2d_3_layer_call_fn_96793р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▓2п
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_96787р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2е
5__inference_batch_normalization_4_layer_call_fn_98466
5__inference_batch_normalization_4_layer_call_fn_98479┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_98499
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_98533┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ї
 __inference__wrapped_model_94640╨@ЫЬЭЮ▒▓Яабв│┤гдеж╡╢зийк╖╕лм╣║онп░EвB
;в8
6К3
module_wrapper_input         dd
к "EкB
@
module_wrapper_23+К(
module_wrapper_23         я
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_98311Ъбв│┤MвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ я
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_98329Ъбв│┤MвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ ╟
5__inference_batch_normalization_1_layer_call_fn_98280Нбв│┤MвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @╟
5__inference_batch_normalization_1_layer_call_fn_98293Нбв│┤MвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @я
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_98373Ъеж╡╢MвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ я
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_98391Ъеж╡╢MвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ ╟
5__inference_batch_normalization_2_layer_call_fn_98342Неж╡╢MвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @╟
5__inference_batch_normalization_2_layer_call_fn_98355Неж╡╢MвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @ё
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_98435Ьйк╖╕NвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ ё
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_98453Ьйк╖╕NвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ╔
5__inference_batch_normalization_3_layer_call_fn_98404Пйк╖╕NвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╔
5__inference_batch_normalization_3_layer_call_fn_98417Пйк╖╕NвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╝
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_98499h╣║он4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ ╝
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_98533h╣║он4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Ф
5__inference_batch_normalization_4_layer_call_fn_98466[╣║он4в1
*в'
!К
inputs         А
p 
к "К         АФ
5__inference_batch_normalization_4_layer_call_fn_98479[╣║он4в1
*в'
!К
inputs         А
p
к "К         Аэ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_98249ЪЭЮ▒▓MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ э
N__inference_batch_normalization_layer_call_and_return_conditional_losses_98267ЪЭЮ▒▓MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ┼
3__inference_batch_normalization_layer_call_fn_98218НЭЮ▒▓MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            ┼
3__inference_batch_normalization_layer_call_fn_98231НЭЮ▒▓MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            э
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_96511ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┼
/__inference_max_pooling2d_1_layer_call_fn_96517СRвO
HвE
CК@
inputs4                                    
к ";К84                                    э
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_96649ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┼
/__inference_max_pooling2d_2_layer_call_fn_96655СRвO
HвE
CК@
inputs4                                    
к ";К84                                    э
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_96787ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┼
/__inference_max_pooling2d_3_layer_call_fn_96793СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ы
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_96373ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ├
-__inference_max_pooling2d_layer_call_fn_96379СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╙
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_97755Веж╡╢GвD
-в*
(К%
args_0         @
к

trainingp "-в*
#К 
0         @
Ъ ╙
L__inference_module_wrapper_10_layer_call_and_return_conditional_losses_97773Веж╡╢GвD
-в*
(К%
args_0         @
к

trainingp"-в*
#К 
0         @
Ъ к
1__inference_module_wrapper_10_layer_call_fn_97724uеж╡╢GвD
-в*
(К%
args_0         @
к

trainingp " К         @к
1__inference_module_wrapper_10_layer_call_fn_97737uеж╡╢GвD
-в*
(К%
args_0         @
к

trainingp" К         @╚
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_97788xGвD
-в*
(К%
args_0         @
к

trainingp "-в*
#К 
0         @
Ъ ╚
L__inference_module_wrapper_11_layer_call_and_return_conditional_losses_97793xGвD
-в*
(К%
args_0         @
к

trainingp"-в*
#К 
0         @
Ъ а
1__inference_module_wrapper_11_layer_call_fn_97778kGвD
-в*
(К%
args_0         @
к

trainingp " К         @а
1__inference_module_wrapper_11_layer_call_fn_97783kGвD
-в*
(К%
args_0         @
к

trainingp" К         @╧
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_97821зиGвD
-в*
(К%
args_0         @
к

trainingp ".в+
$К!
0         А
Ъ ╧
L__inference_module_wrapper_12_layer_call_and_return_conditional_losses_97831зиGвD
-в*
(К%
args_0         @
к

trainingp".в+
$К!
0         А
Ъ з
1__inference_module_wrapper_12_layer_call_fn_97802rзиGвD
-в*
(К%
args_0         @
к

trainingp "!К         Аз
1__inference_module_wrapper_12_layer_call_fn_97811rзиGвD
-в*
(К%
args_0         @
к

trainingp"!К         А╩
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_97846zHвE
.в+
)К&
args_0         А
к

trainingp ".в+
$К!
0         А
Ъ ╩
L__inference_module_wrapper_13_layer_call_and_return_conditional_losses_97851zHвE
.в+
)К&
args_0         А
к

trainingp".в+
$К!
0         А
Ъ в
1__inference_module_wrapper_13_layer_call_fn_97836mHвE
.в+
)К&
args_0         А
к

trainingp "!К         Ав
1__inference_module_wrapper_13_layer_call_fn_97841mHвE
.в+
)К&
args_0         А
к

trainingp"!К         А╒
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_97895Дйк╖╕HвE
.в+
)К&
args_0         А
к

trainingp ".в+
$К!
0         А
Ъ ╒
L__inference_module_wrapper_14_layer_call_and_return_conditional_losses_97913Дйк╖╕HвE
.в+
)К&
args_0         А
к

trainingp".в+
$К!
0         А
Ъ м
1__inference_module_wrapper_14_layer_call_fn_97864wйк╖╕HвE
.в+
)К&
args_0         А
к

trainingp "!К         Ам
1__inference_module_wrapper_14_layer_call_fn_97877wйк╖╕HвE
.в+
)К&
args_0         А
к

trainingp"!К         А╩
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_97928zHвE
.в+
)К&
args_0         А
к

trainingp ".в+
$К!
0         А
Ъ ╩
L__inference_module_wrapper_15_layer_call_and_return_conditional_losses_97933zHвE
.в+
)К&
args_0         А
к

trainingp".в+
$К!
0         А
Ъ в
1__inference_module_wrapper_15_layer_call_fn_97918mHвE
.в+
)К&
args_0         А
к

trainingp "!К         Ав
1__inference_module_wrapper_15_layer_call_fn_97923mHвE
.в+
)К&
args_0         А
к

trainingp"!К         А╩
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_97948zHвE
.в+
)К&
args_0         А
к

trainingp ".в+
$К!
0         А
Ъ ╩
L__inference_module_wrapper_16_layer_call_and_return_conditional_losses_97960zHвE
.в+
)К&
args_0         А
к

trainingp".в+
$К!
0         А
Ъ в
1__inference_module_wrapper_16_layer_call_fn_97938mHвE
.в+
)К&
args_0         А
к

trainingp "!К         Ав
1__inference_module_wrapper_16_layer_call_fn_97943mHвE
.в+
)К&
args_0         А
к

trainingp"!К         А┬
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_97976rHвE
.в+
)К&
args_0         А
к

trainingp "&в#
К
0         А
Ъ ┬
L__inference_module_wrapper_17_layer_call_and_return_conditional_losses_97982rHвE
.в+
)К&
args_0         А
к

trainingp"&в#
К
0         А
Ъ Ъ
1__inference_module_wrapper_17_layer_call_fn_97965eHвE
.в+
)К&
args_0         А
к

trainingp "К         АЪ
1__inference_module_wrapper_17_layer_call_fn_97970eHвE
.в+
)К&
args_0         А
к

trainingp"К         А└
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_98010pлм@в=
&в#
!К
args_0         А
к

trainingp "&в#
К
0         А
Ъ └
L__inference_module_wrapper_18_layer_call_and_return_conditional_losses_98020pлм@в=
&в#
!К
args_0         А
к

trainingp"&в#
К
0         А
Ъ Ш
1__inference_module_wrapper_18_layer_call_fn_97991cлм@в=
&в#
!К
args_0         А
к

trainingp "К         АШ
1__inference_module_wrapper_18_layer_call_fn_98000cлм@в=
&в#
!К
args_0         А
к

trainingp"К         А║
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_98035j@в=
&в#
!К
args_0         А
к

trainingp "&в#
К
0         А
Ъ ║
L__inference_module_wrapper_19_layer_call_and_return_conditional_losses_98040j@в=
&в#
!К
args_0         А
к

trainingp"&в#
К
0         А
Ъ Т
1__inference_module_wrapper_19_layer_call_fn_98025]@в=
&в#
!К
args_0         А
к

trainingp "К         АТ
1__inference_module_wrapper_19_layer_call_fn_98030]@в=
&в#
!К
args_0         А
к

trainingp"К         А╟
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_97426xGвD
-в*
(К%
args_0         dd 
к

trainingp "-в*
#К 
0         dd 
Ъ ╟
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_97431xGвD
-в*
(К%
args_0         dd 
к

trainingp"-в*
#К 
0         dd 
Ъ Я
0__inference_module_wrapper_1_layer_call_fn_97416kGвD
-в*
(К%
args_0         dd 
к

trainingp " К         dd Я
0__inference_module_wrapper_1_layer_call_fn_97421kGвD
-в*
(К%
args_0         dd 
к

trainingp" К         dd ─
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_98086t╣║он@в=
&в#
!К
args_0         А
к

trainingp "&в#
К
0         А
Ъ ─
L__inference_module_wrapper_20_layer_call_and_return_conditional_losses_98120t╣║он@в=
&в#
!К
args_0         А
к

trainingp"&в#
К
0         А
Ъ Ь
1__inference_module_wrapper_20_layer_call_fn_98053g╣║он@в=
&в#
!К
args_0         А
к

trainingp "К         АЬ
1__inference_module_wrapper_20_layer_call_fn_98066g╣║он@в=
&в#
!К
args_0         А
к

trainingp"К         А║
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_98135j@в=
&в#
!К
args_0         А
к

trainingp "&в#
К
0         А
Ъ ║
L__inference_module_wrapper_21_layer_call_and_return_conditional_losses_98147j@в=
&в#
!К
args_0         А
к

trainingp"&в#
К
0         А
Ъ Т
1__inference_module_wrapper_21_layer_call_fn_98125]@в=
&в#
!К
args_0         А
к

trainingp "К         АТ
1__inference_module_wrapper_21_layer_call_fn_98130]@в=
&в#
!К
args_0         А
к

trainingp"К         А┐
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_98175oп░@в=
&в#
!К
args_0         А
к

trainingp "%в"
К
0         
Ъ ┐
L__inference_module_wrapper_22_layer_call_and_return_conditional_losses_98185oп░@в=
&в#
!К
args_0         А
к

trainingp"%в"
К
0         
Ъ Ч
1__inference_module_wrapper_22_layer_call_fn_98156bп░@в=
&в#
!К
args_0         А
к

trainingp "К         Ч
1__inference_module_wrapper_22_layer_call_fn_98165bп░@в=
&в#
!К
args_0         А
к

trainingp"К         ╕
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_98200h?в<
%в"
 К
args_0         
к

trainingp "%в"
К
0         
Ъ ╕
L__inference_module_wrapper_23_layer_call_and_return_conditional_losses_98205h?в<
%в"
 К
args_0         
к

trainingp"%в"
К
0         
Ъ Р
1__inference_module_wrapper_23_layer_call_fn_98190[?в<
%в"
 К
args_0         
к

trainingp "К         Р
1__inference_module_wrapper_23_layer_call_fn_98195[?в<
%в"
 К
args_0         
к

trainingp"К         ╥
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_97475ВЭЮ▒▓GвD
-в*
(К%
args_0         dd 
к

trainingp "-в*
#К 
0         dd 
Ъ ╥
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_97493ВЭЮ▒▓GвD
-в*
(К%
args_0         dd 
к

trainingp"-в*
#К 
0         dd 
Ъ й
0__inference_module_wrapper_2_layer_call_fn_97444uЭЮ▒▓GвD
-в*
(К%
args_0         dd 
к

trainingp " К         dd й
0__inference_module_wrapper_2_layer_call_fn_97457uЭЮ▒▓GвD
-в*
(К%
args_0         dd 
к

trainingp" К         dd ╟
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_97508xGвD
-в*
(К%
args_0         dd 
к

trainingp "-в*
#К 
0         !! 
Ъ ╟
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_97513xGвD
-в*
(К%
args_0         dd 
к

trainingp"-в*
#К 
0         !! 
Ъ Я
0__inference_module_wrapper_3_layer_call_fn_97498kGвD
-в*
(К%
args_0         dd 
к

trainingp " К         !! Я
0__inference_module_wrapper_3_layer_call_fn_97503kGвD
-в*
(К%
args_0         dd 
к

trainingp" К         !! ═
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_97541~ЯаGвD
-в*
(К%
args_0         !! 
к

trainingp "-в*
#К 
0         !!@
Ъ ═
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_97551~ЯаGвD
-в*
(К%
args_0         !! 
к

trainingp"-в*
#К 
0         !!@
Ъ е
0__inference_module_wrapper_4_layer_call_fn_97522qЯаGвD
-в*
(К%
args_0         !! 
к

trainingp " К         !!@е
0__inference_module_wrapper_4_layer_call_fn_97531qЯаGвD
-в*
(К%
args_0         !! 
к

trainingp" К         !!@╟
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_97566xGвD
-в*
(К%
args_0         !!@
к

trainingp "-в*
#К 
0         !!@
Ъ ╟
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_97571xGвD
-в*
(К%
args_0         !!@
к

trainingp"-в*
#К 
0         !!@
Ъ Я
0__inference_module_wrapper_5_layer_call_fn_97556kGвD
-в*
(К%
args_0         !!@
к

trainingp " К         !!@Я
0__inference_module_wrapper_5_layer_call_fn_97561kGвD
-в*
(К%
args_0         !!@
к

trainingp" К         !!@╥
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_97615Вбв│┤GвD
-в*
(К%
args_0         !!@
к

trainingp "-в*
#К 
0         !!@
Ъ ╥
K__inference_module_wrapper_6_layer_call_and_return_conditional_losses_97633Вбв│┤GвD
-в*
(К%
args_0         !!@
к

trainingp"-в*
#К 
0         !!@
Ъ й
0__inference_module_wrapper_6_layer_call_fn_97584uбв│┤GвD
-в*
(К%
args_0         !!@
к

trainingp " К         !!@й
0__inference_module_wrapper_6_layer_call_fn_97597uбв│┤GвD
-в*
(К%
args_0         !!@
к

trainingp" К         !!@╟
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_97648xGвD
-в*
(К%
args_0         !!@
к

trainingp "-в*
#К 
0         @
Ъ ╟
K__inference_module_wrapper_7_layer_call_and_return_conditional_losses_97653xGвD
-в*
(К%
args_0         !!@
к

trainingp"-в*
#К 
0         @
Ъ Я
0__inference_module_wrapper_7_layer_call_fn_97638kGвD
-в*
(К%
args_0         !!@
к

trainingp " К         @Я
0__inference_module_wrapper_7_layer_call_fn_97643kGвD
-в*
(К%
args_0         !!@
к

trainingp" К         @═
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_97681~гдGвD
-в*
(К%
args_0         @
к

trainingp "-в*
#К 
0         @
Ъ ═
K__inference_module_wrapper_8_layer_call_and_return_conditional_losses_97691~гдGвD
-в*
(К%
args_0         @
к

trainingp"-в*
#К 
0         @
Ъ е
0__inference_module_wrapper_8_layer_call_fn_97662qгдGвD
-в*
(К%
args_0         @
к

trainingp " К         @е
0__inference_module_wrapper_8_layer_call_fn_97671qгдGвD
-в*
(К%
args_0         @
к

trainingp" К         @╟
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_97706xGвD
-в*
(К%
args_0         @
к

trainingp "-в*
#К 
0         @
Ъ ╟
K__inference_module_wrapper_9_layer_call_and_return_conditional_losses_97711xGвD
-в*
(К%
args_0         @
к

trainingp"-в*
#К 
0         @
Ъ Я
0__inference_module_wrapper_9_layer_call_fn_97696kGвD
-в*
(К%
args_0         @
к

trainingp " К         @Я
0__inference_module_wrapper_9_layer_call_fn_97701kGвD
-в*
(К%
args_0         @
к

trainingp" К         @╦
I__inference_module_wrapper_layer_call_and_return_conditional_losses_97401~ЫЬGвD
-в*
(К%
args_0         dd
к

trainingp "-в*
#К 
0         dd 
Ъ ╦
I__inference_module_wrapper_layer_call_and_return_conditional_losses_97411~ЫЬGвD
-в*
(К%
args_0         dd
к

trainingp"-в*
#К 
0         dd 
Ъ г
.__inference_module_wrapper_layer_call_fn_97382qЫЬGвD
-в*
(К%
args_0         dd
к

trainingp " К         dd г
.__inference_module_wrapper_layer_call_fn_97391qЫЬGвD
-в*
(К%
args_0         dd
к

trainingp" К         dd В
E__inference_sequential_layer_call_and_return_conditional_losses_96074╕@ЫЬЭЮ▒▓Яабв│┤гдеж╡╢зийк╖╕лм╣║онп░MвJ
Cв@
6К3
module_wrapper_input         dd
p 

 
к "%в"
К
0         
Ъ В
E__inference_sequential_layer_call_and_return_conditional_losses_96166╕@ЫЬЭЮ▒▓Яабв│┤гдеж╡╢зийк╖╕лм╣║онп░MвJ
Cв@
6К3
module_wrapper_input         dd
p

 
к "%в"
К
0         
Ъ Ї
E__inference_sequential_layer_call_and_return_conditional_losses_97219к@ЫЬЭЮ▒▓Яабв│┤гдеж╡╢зийк╖╕лм╣║онп░?в<
5в2
(К%
inputs         dd
p 

 
к "%в"
К
0         
Ъ Ї
E__inference_sequential_layer_call_and_return_conditional_losses_97373к@ЫЬЭЮ▒▓Яабв│┤гдеж╡╢зийк╖╕лм╣║онп░?в<
5в2
(К%
inputs         dd
p

 
к "%в"
К
0         
Ъ ┌
*__inference_sequential_layer_call_fn_95045л@ЫЬЭЮ▒▓Яабв│┤гдеж╡╢зийк╖╕лм╣║онп░MвJ
Cв@
6К3
module_wrapper_input         dd
p 

 
к "К         ┌
*__inference_sequential_layer_call_fn_95982л@ЫЬЭЮ▒▓Яабв│┤гдеж╡╢зийк╖╕лм╣║онп░MвJ
Cв@
6К3
module_wrapper_input         dd
p

 
к "К         ╠
*__inference_sequential_layer_call_fn_97024Э@ЫЬЭЮ▒▓Яабв│┤гдеж╡╢зийк╖╕лм╣║онп░?в<
5в2
(К%
inputs         dd
p 

 
к "К         ╠
*__inference_sequential_layer_call_fn_97093Э@ЫЬЭЮ▒▓Яабв│┤гдеж╡╢зийк╖╕лм╣║онп░?в<
5в2
(К%
inputs         dd
p

 
к "К         Р
#__inference_signature_wrapper_96241ш@ЫЬЭЮ▒▓Яабв│┤гдеж╡╢зийк╖╕лм╣║онп░]вZ
в 
SкP
N
module_wrapper_input6К3
module_wrapper_input         dd"EкB
@
module_wrapper_23+К(
module_wrapper_23         