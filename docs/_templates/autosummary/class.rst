{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :exclude-members: model_post_init, model_fields, model_computed_fields, model_config, get_static, prepare

{% block methods %}{% endblock %}

{% block attributes %}{% endblock %}
