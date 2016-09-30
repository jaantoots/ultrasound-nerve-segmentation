# Identify nerve structures in ultrasound images of the neck

This is my solution to the Kaggle Ultrasound Nerve Segmentation
competition using a VGG net for semantic segmentation implemented in
Torch.

Documentation for training and testing:

```shell
th train.lua --help
th test.lua --help
```

The model is described in `model.lua`.

## Description

Even the bravest patient cringes at the mention of a surgical
procedure. Surgery inevitably brings discomfort, and oftentimes
involves significant post-surgical pain. Currently, patient pain is
frequently managed through the use of narcotics that bring a bevy of
unwanted side effects.

This competition's sponsor is working to improve pain management
through the use of indwelling catheters that block or mitigate pain at
the source. Pain management catheters reduce dependence on narcotics
and speed up patient recovery.

Accurately identifying nerve structures in ultrasound images is a
critical step in effectively inserting a patient’s pain management
catheter. In this competition, Kagglers are challenged to build a
model that can identify nerve structures in a dataset of ultrasound
images of the neck. Doing so would improve catheter placement and
contribute to a more pain free future.

[Kaggle competition](https://www.kaggle.com/c/ultrasound-nerve-segmentation)

## License

Copyright (C) 2016  Jaan Toots

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
