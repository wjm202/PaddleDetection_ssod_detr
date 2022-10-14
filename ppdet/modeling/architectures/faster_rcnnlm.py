# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register
from .faster_rcnn import FasterRCNN

__all__ = ['FasterRCNNLM']


@register
class FasterRCNNLM(FasterRCNN):
    """
    Faster R-CNN network, see https://arxiv.org/abs/1506.01497

    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNHead` instance
        bbox_head (object): `BBoxHead` instance
        bbox_post_process (object): `BBoxPostProcess` instance
        neck (object): 'FPN' instance
    """
    __category__ = 'architecture'
    __inject__ = ['bbox_post_process']

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 bbox_post_process,
                 neck=None):
        super(FasterRCNNLM, self).__init__(backbone, rpn_head, bbox_head,
                                            bbox_post_process, neck)