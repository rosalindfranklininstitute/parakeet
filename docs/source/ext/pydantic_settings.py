#
# pydantic_settings.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import importlib
import os.path
from docutils import nodes
from docutils.parsers.rst import Directive


class PydanticDirective(Directive):
    """
    A sphinx plugin to pull out configuration information from pydantic

    """

    has_content = True

    def run(self):

        # Get the pydantic schema
        assert len(self.content) == 1
        schema = self.from_import(self.content[0])

        # Get the document nodes
        return self.nodes_from_schema(schema)

    def nodes_from_schema(self, schema):

        # Walk through objects
        def walk(obj):

            # Dictionaries are made into field_lists types
            if isinstance(obj, dict):

                paragraph = nodes.field_list()

                for key, value in obj.items():

                    if key in ["default", "definitions"]:
                        continue
                    elif key in ["$ref"]:
                        value = os.path.basename(value)
                        p = nodes.paragraph()
                        p += nodes.reference(
                            text=value, refuri="#%s" % nodes.make_id(value)
                        )
                    elif isinstance(value, dict):
                        p = walk(value)
                    elif isinstance(value, list):
                        p = walk(value)
                    else:
                        p = nodes.paragraph(text=value)

                    paragraph += nodes.field(
                        "", nodes.field_name(text=key), nodes.field_body("", p)
                    )

            # Lists are made into bullet_list types
            elif isinstance(obj, list):

                paragraph = nodes.bullet_list("")

                for value in obj:

                    if isinstance(value, dict):
                        p = walk(value)
                    elif isinstance(value, list):
                        p = walk(value)
                    else:
                        p = nodes.paragraph(text=value)

                    paragraph += nodes.list_item("", p)

            return paragraph

        # Parse the top level
        toplevel = walk(schema)

        # Parse the definitions
        definitions = nodes.paragraph()
        for key, value in schema["$defs"].items():
            definition = nodes.section(ids=[nodes.make_id(key)])
            definition += nodes.title(text=key)
            definition += walk(value)
            definitions += definition
        return [toplevel, definitions]

    def from_import(self, loc):

        # Parse the first argument, the import statement for pydantic object
        package_items = loc.split(".")
        package_name = package_items[0]
        module_name = ".".join(package_items[0:-1])
        obj_name = package_items[-1]

        # Try importing object
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, obj_name)
        except ImportError as e:
            raise e

        # Get the schema
        return obj.model_json_schema()


def setup(app):
    """
    Setup the extension

    The extension can be used as follows:

    .. pydantic:: parakeet.config.Config

    """
    app.add_directive("pydantic", PydanticDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
