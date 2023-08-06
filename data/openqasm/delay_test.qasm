OPENQASM 2.0;
include "qelib1.inc";
opaque delay(param0) q0;
qreg q[2];
// h q[0]
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];

delay(4e-6) q[1];
cx q[0],q[1];
