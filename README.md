# mRAG Fashion RAG

## System design
High-level flow and diagrams are available in [Asset/system design.pdf](Asset/system%20design.pdf).

## Here are some test case to help better illustration of the system: 
### 1) Text search -> outfit pairing
- Query example: User want to find a blue jacket for outdoor activity with some additional requirement 
- Input image: ![Test blue jacket input](Asset/Test_blue_jacket.png)
- Pairing result: ![Blue jacket shoe pairing](Asset/Test_blue_jacket_shoe_pairing.png)

### 2) Image search using ref image -> color variant
- In this case, the user see a hoodie somewhere and want to find the same thing, and then he want to find black color version of the hoodie
- Reference image search: ![Search by image input](Asset/Test_search_by_img.png)
- Color variant result: ![Search by image color variant](Asset/Test_search_by_img_color_variant.png)

## Assets
- UI snapshot: ![UI snapshot](Asset/UI.png)