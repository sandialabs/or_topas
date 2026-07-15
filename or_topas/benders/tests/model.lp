\* Source Pyomo model name=unknown *\

min 
obj:
+2.0 generation(1)
+2.0 generation(2)
+2.0 generation(3)
+2.0 generation(4)
+50.0 generation(5)

s.t.

c_e_FlowBalance(1)_:
-1 flows(1_2)
-1 flows(1_4)
-1 flows(1_5)
+1 generation(1)
= 0

c_e_FlowBalance(2)_:
+1 flows(1_2)
-1 flows(2_3)
+1 generation(2)
= 3.0

c_e_FlowBalance(3)_:
+1 flows(2_3)
-1 flows(3_4)
+1 generation(3)
= 3.0

c_e_FlowBalance(4)_:
+1 flows(1_4)
+1 flows(3_4)
-1 flows(4_5)
+1 generation(4)
= 4.0

c_e_FlowBalance(5)_:
+1 flows(1_5)
+1 flows(4_5)
+1 generation(5)
= 0

c_e_PowerBalance(1_2)_:
+35.587188612099645 angles(1)
-35.587188612099645 angles(2)
+1 flows(1_2)
= 0

c_e_PowerBalance(1_4)_:
+32.89473684210526 angles(1)
-32.89473684210526 angles(4)
+1 flows(1_4)
= 0

c_e_PowerBalance(1_5)_:
+156.25 angles(1)
-156.25 angles(5)
+1 flows(1_5)
= 0

c_e_PowerBalance(2_3)_:
+92.59259259259258 angles(2)
-92.59259259259258 angles(3)
+1 flows(2_3)
= 0

c_e_PowerBalance(3_4)_:
+33.67003367003367 angles(3)
-33.67003367003367 angles(4)
+1 flows(3_4)
= 0

c_e_PowerBalance(4_5)_:
+33.67003367003367 angles(4)
-33.67003367003367 angles(5)
+1 flows(4_5)
= 0

c_l_angles_lower(1)_:
+1 angles(1)
>= -3.141592653589793

c_l_angles_lower(2)_:
+1 angles(2)
>= -3.141592653589793

c_l_angles_lower(3)_:
+1 angles(3)
>= -3.141592653589793

c_l_angles_lower(4)_:
+1 angles(4)
>= -3.141592653589793

c_l_angles_lower(5)_:
+1 angles(5)
>= -3.141592653589793

c_u_angles_upper(1)_:
+1 angles(1)
<= 3.141592653589793

c_u_angles_upper(2)_:
+1 angles(2)
<= 3.141592653589793

c_u_angles_upper(3)_:
+1 angles(3)
<= 3.141592653589793

c_u_angles_upper(4)_:
+1 angles(4)
<= 3.141592653589793

c_u_angles_upper(5)_:
+1 angles(5)
<= 3.141592653589793

c_l_flow_lower(1_2)_:
+1 flows(1_2)
>= -4.0

c_l_flow_lower(1_4)_:
+1 flows(1_4)
>= -4.26

c_l_flow_lower(1_5)_:
+1 flows(1_5)
>= -4.26

c_l_flow_lower(2_3)_:
+1 flows(2_3)
>= -4.26

c_l_flow_lower(3_4)_:
+1 flows(3_4)
>= -4.26

c_l_flow_lower(4_5)_:
+1 flows(4_5)
>= -2.4

c_u_flow_upper(1_2)_:
+1 flows(1_2)
<= 4.0

c_u_flow_upper(1_4)_:
+1 flows(1_4)
<= 4.26

c_u_flow_upper(1_5)_:
+1 flows(1_5)
<= 4.26

c_u_flow_upper(2_3)_:
+1 flows(2_3)
<= 4.26

c_u_flow_upper(3_4)_:
+1 flows(3_4)
<= 4.26

c_u_flow_upper(4_5)_:
+1 flows(4_5)
<= 2.4

c_l_gen_lower(1)_:
+1 generation(1)
>= 0

c_l_gen_lower(2)_:
+1 generation(2)
>= 0

c_l_gen_lower(3)_:
+1 generation(3)
>= 0

c_l_gen_lower(4)_:
+1 generation(4)
>= 0

c_l_gen_lower(5)_:
+1 generation(5)
>= 0

c_u_gen_upper(1)_:
+1 generation(1)
<= 2.1

c_u_gen_upper(2)_:
+1 generation(2)
<= 0.0

c_u_gen_upper(3)_:
+1 generation(3)
<= 5.2

c_u_gen_upper(4)_:
+1 generation(4)
<= 2.0

c_u_gen_upper(5)_:
+1 generation(5)
<= 6.0

bounds
   -4.141592653589793 <= angles(1) <= 4.141592653589793
   -4.141592653589793 <= angles(2) <= 4.141592653589793
   -4.141592653589793 <= angles(3) <= 4.141592653589793
   -4.141592653589793 <= angles(4) <= 4.141592653589793
   -4.141592653589793 <= angles(5) <= 4.141592653589793
   -5.0 <= flows(1_2) <= 5.0
   -5.26 <= flows(1_4) <= 5.26
   -5.26 <= flows(1_5) <= 5.26
   -5.26 <= flows(2_3) <= 5.26
   -5.26 <= flows(3_4) <= 5.26
   -3.4 <= flows(4_5) <= 3.4
   0 <= generation(1) <= 3.1
   0 <= generation(2) <= 1.0
   0 <= generation(3) <= 6.2
   0 <= generation(4) <= 3.0
   0 <= generation(5) <= 7.0
end
