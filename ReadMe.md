So whats the data
---- 


Input:
Study > Series > 'Sagittal T2/STIR', 'Sagittal T1', 'Axial T2' (per person) 

Img shapes:
Sagittal T1 -> (640, 640)

Axial T2 ->  (256, 256)

Sagittal T2/STIR -> (768, 768)




Ideas:
Merge Study and Series key


Output:
Study > Series > 


Model:
One model for each 'Sagittal T2/STIR', 'Sagittal T1', 'Axial T2' 