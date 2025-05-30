OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg meas[21];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(3*pi/4) q[1];
sx q[1];
cx q[0],q[1];
sx q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(-pi) q[1];
cx q[0],q[1];
rz(-pi) q[2];
sx q[2];
rz(2.526112944919406) q[2];
sx q[2];
cx q[1],q[2];
sx q[2];
rz(2.526112944919406) q[2];
sx q[2];
rz(-pi) q[2];
cx q[1],q[2];
rz(-pi) q[3];
sx q[3];
rz(5*pi/6) q[3];
sx q[3];
cx q[2],q[3];
sx q[3];
rz(5*pi/6) q[3];
sx q[3];
rz(-pi) q[3];
cx q[2],q[3];
rz(-pi) q[4];
sx q[4];
rz(2.677945044588988) q[4];
sx q[4];
cx q[3],q[4];
sx q[4];
rz(2.677945044588987) q[4];
sx q[4];
rz(-pi) q[4];
cx q[3],q[4];
rz(-pi) q[5];
sx q[5];
rz(2.721058318305828) q[5];
sx q[5];
cx q[4],q[5];
sx q[5];
rz(2.721058318305828) q[5];
sx q[5];
rz(-pi) q[5];
cx q[4],q[5];
rz(-pi) q[6];
sx q[6];
rz(2.753995966934612) q[6];
sx q[6];
cx q[5],q[6];
sx q[6];
rz(2.753995966934612) q[6];
sx q[6];
rz(-pi) q[6];
cx q[5],q[6];
rz(-pi) q[7];
sx q[7];
rz(2.7802255296830856) q[7];
sx q[7];
cx q[6],q[7];
sx q[7];
rz(2.7802255296830856) q[7];
sx q[7];
rz(-pi) q[7];
cx q[6],q[7];
rz(-pi) q[8];
sx q[8];
rz(2.8017557441356713) q[8];
sx q[8];
cx q[7],q[8];
sx q[8];
rz(2.8017557441356713) q[8];
sx q[8];
rz(-pi) q[8];
cx q[7],q[8];
rz(-pi) q[9];
sx q[9];
rz(2.819842099193151) q[9];
sx q[9];
cx q[8],q[9];
sx q[9];
rz(2.819842099193151) q[9];
sx q[9];
rz(-pi) q[9];
cx q[8],q[9];
rz(-pi) q[10];
sx q[10];
rz(2.8353152844201235) q[10];
sx q[10];
cx q[9],q[10];
sx q[10];
rz(2.8353152844201226) q[10];
sx q[10];
rz(-pi) q[10];
cx q[9],q[10];
rz(-pi) q[11];
sx q[11];
rz(2.8487498818612176) q[11];
sx q[11];
cx q[10],q[11];
sx q[11];
rz(2.8487498818612176) q[11];
sx q[11];
rz(-pi) q[11];
cx q[10],q[11];
rz(-pi) q[12];
sx q[12];
rz(2.8605577520869794) q[12];
sx q[12];
cx q[11],q[12];
sx q[12];
rz(2.8605577520869794) q[12];
sx q[12];
rz(-pi) q[12];
cx q[11],q[12];
rz(-pi) q[13];
sx q[13];
rz(2.8710428906112204) q[13];
sx q[13];
cx q[12],q[13];
sx q[13];
rz(2.8710428906112204) q[13];
sx q[13];
rz(-pi) q[13];
cx q[12],q[13];
rz(-pi) q[14];
sx q[14];
rz(2.8804352426867688) q[14];
sx q[14];
cx q[13],q[14];
sx q[14];
rz(2.8804352426867688) q[14];
sx q[14];
rz(-pi) q[14];
cx q[13],q[14];
rz(-pi) q[15];
sx q[15];
rz(2.8889123984477143) q[15];
sx q[15];
cx q[14],q[15];
sx q[15];
rz(2.8889123984477143) q[15];
sx q[15];
rz(-pi) q[15];
cx q[14],q[15];
rz(-pi) q[16];
sx q[16];
rz(2.896613990462929) q[16];
sx q[16];
cx q[15],q[16];
sx q[16];
rz(2.896613990462929) q[16];
sx q[16];
rz(-pi) q[16];
cx q[15],q[16];
rz(-pi) q[17];
sx q[17];
rz(2.9036515287595854) q[17];
sx q[17];
cx q[16],q[17];
sx q[17];
rz(2.9036515287595854) q[17];
sx q[17];
rz(-pi) q[17];
cx q[16],q[17];
rz(-pi) q[18];
sx q[18];
rz(2.9101152896196147) q[18];
sx q[18];
cx q[17],q[18];
sx q[18];
rz(2.9101152896196147) q[18];
sx q[18];
rz(-pi) q[18];
cx q[17],q[18];
rz(-pi) q[19];
sx q[19];
rz(2.9160792476916617) q[19];
sx q[19];
cx q[18],q[19];
sx q[19];
rz(2.9160792476916617) q[19];
sx q[19];
rz(-pi) q[19];
cx q[18],q[19];
rz(-pi) q[20];
sx q[20];
rz(2.9216046761943337) q[20];
sx q[20];
cx q[19],q[20];
sx q[20];
rz(2.9216046761943337) q[20];
sx q[20];
rz(-pi) q[20];
cx q[19],q[20];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
measure q[16] -> meas[16];
measure q[17] -> meas[17];
measure q[18] -> meas[18];
measure q[19] -> meas[19];
measure q[20] -> meas[20];