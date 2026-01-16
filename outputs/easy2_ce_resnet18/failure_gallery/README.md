# Failure Gallery

- **171215509_5015d8f400.jpg** | true: `pizza` → pred: `pancakes` (conf=0.99)
  - Diagnosis: rectangular pizza slices lack common ‘round pizza’ cues; heavily browned cheese and flat shape resemble batter after resizing

- **64905204_6106fcf433.jpg** | true: `cheesecake` → pred: `cup_cakes` (conf=0.99)
  - Diagnosis: cupcake-like size and presentation of mini cheesecakes similar to how cupcakes are presented, caused reliance on shape and portion

- **77846587_424d82808c.jpg** | true: `tiramisu` → pred: `chocolate_cake` (conf=0.97)
  - Diagnosis: the thick cocoa layer and cake-like shape make it easy to mistake for chocolate cake

- **79217877_b5a0a82fc8.jpg** | true: `cheesecake` → pred: `cheese_plate` (conf=0.95)
  - Diagnosis: neat slices and uniform pale color resemble a cheese plate more than a dessert

- **24281615_fed281e0fe.jpg** | true: `pizza` → pred: `seaweed_salad` (conf=0.95)
  - Diagnosis: a large amount of greens covers the pizza, so the model focuses on salad-like textures instead of the crust

- **80212085_bfadb4c470.jpg** | true: `sushi` → pred: `sashimi` (conf=0.95)
  - Diagnosis: mostly on this plate are raw fish slices, which makes it look more like sashimi than sushi

- **158682415_a5dc948382.jpg** | true: `sushi` → pred: `pancakes` (conf=0.94)
  - Diagnosis: close-up removes context, making the flat brown-ish texture resemble pancakes instead of sushi

- **165143737_2f09385209.jpg** | true: `ice_cream` → pred: `cup_cakes` (conf=0.93)
  - Diagnosis: swirled topping and paper-wrapped cones look similar to cupcake frosting

- **51257467_fb5cd93563.jpg** | true: `caesar_salad` → pred: `frozen_yogurt` (conf=0.93)
  - Diagnosis: only packaging is visible, not the actual food

- **129113287_139f4adaef.jpg** | true: `tiramisu` → pred: `pancakes` (conf=0.92)
  - Diagnosis: flat, layered dessert in a rectangular dish looks more like stacked pancakes at this angle

- **40949822_cdc4c9bcfc.jpg** | true: `ice_cream` → pred: `frozen_yogurt` (conf=0.91)
  - Diagnosis: served in a bowl with fruit toppings, which looks more like frozen yogurt than ice cream

- **14319296_abd5c7868f.jpg** | true: `paella` → pred: `seaweed_salad` (conf=0.91)
  - Diagnosis: paella is usually shown in a pan; greenish color and chopped leafy texture make it look more like a salad

- **30371139_93f4900b7e.jpg** | true: `cheesecake` → pred: `macarons` (conf=0.90)
  - Diagnosis: image shows ingredients rather than the finished cheesecake

- **57019563_4ce96782bd.jpg** | true: `ice_cream` → pred: `frozen_yogurt` (conf=0.90)
  - Diagnosis: shown in a cup, which makes it resemble frozen yogurt

- **96003794_6186e88a4d.jpg** | true: `sushi` → pred: `cheese_plate` (conf=0.89)
  - Diagnosis: neatly arranged slices makes it look like a cheese plate

Most failures are driven by presentation bias and context loss rather than semantic misunderstanding. The model often relies on shape, color, and serving style and struggles when typical class cues are missing or unshown. Several errors also highlight dataset noise (ingredients-only images, packaging) and domain shift from canonical Food-101 presentations to real-world photos. Many misclassifications occur with very high confidence, reinforcing the need for improved calibration and out-of-distribution awareness.