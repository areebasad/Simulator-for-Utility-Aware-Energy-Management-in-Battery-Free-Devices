from typing import Dict, List, Any, Optional
import pandas as pd
from abc import abstractmethod

from enum import Enum


class VariableMode(Enum):
    LOADED = 1
    CALCULATED = 2
    DERIVED = 3


class Component:
    def __init__(
        self, config: Dict[str, Any], type: str, label: Optional[str] = None
    ) -> None:
        self.config: Dict[str, Any] = config
        self.type: str = type
        self.label: Optional[str] = label
        self.sub_components: List[Component] = []
        self.parameters: List[Dict[str, Any]] = []
        self.variables: List[Dict[str, Any]] = []

    def add_component(self, component: "Component") -> None:
        self.sub_components.append(component)

    @abstractmethod
    def init_dataframe(self, df: pd.DataFrame) -> None:
        """Overwrite this method to initializa the dataframe."""
        ...

    def init_all_components(self, df: pd.DataFrame) -> None:
        """Per default, calls init_dataframe on this component and all its subcomponents, in the order the components were added.
        Override if the sequence matters."""
        return self.init_dataframe(df)
        
        #for component in self.sub_components:
        #    component.init_dataframe(df)

    def add_parameter(
        self, key: str, value: Any, unit: str = "", derived: bool = False
    ) -> None:
        """A parameter is any type of value of a component that does not change within a simulation run."""
        self.parameters.append(
            {"key": key, "value": value, "unit": unit, "derived": derived}
        )

    def add_variable(
        self,
        df: pd.DataFrame,
        name: str,
        mode: VariableMode,
        unit: str = "",
        default: Optional[Any] = None,
        title: Optional[str] = None,
    ) -> None:
        """A variable that is traced through simulation.
        mode: loaded, calculated, derived.

        Initializes the column of a dataframe for the variable.
        Should be called by subclasses when they override init_dataframe.
        """
        self.variables.append(
            {
                "name": name,
                "title": title if title else name,
                "unit": unit,
                "default": default,
                "mode": mode,
            }
        )
        df[name] = default

    def _properties_html(self) -> str:
        html: List[str] = []
        html.append("<table>")
        for p in self.parameters:
            html.append(
                f'<tr><td>{p["key"]}</td><td>{p["value"]}</td><td>{p["unit"]}</td></tr>'
            )
        for v in self.variables:
            html.append(
                f'<tr><td>{v["key"]}</td><td>(simulated)</td><td>{v["unit"]}</td></tr>'
            )
        html.append("</table>")
        return "\n".join(html)

    def _repr_html_(self) -> str:
        # if it were a dict:
        # i = pd.DataFrame(self.config.values(), index=self.config.keys(), columns=['value'])
        # but config is a series:
        html: List[str] = []
        html.append('<table style="font-size: xx-small; text-align: left">')
        if self.label is not None:
            html.append(
                f'<tr style="border-top: 1px solid black; background-color:darkgray; color: white; font-weight:normal"><th colspan="4">{self.label}</th>'
            )

        def append_component(html: List[str], component: "Component", title: str):
            first = True
            for p in component.parameters:
                style = (
                    "background: white !important; color: #a9a9a9"
                    if (p["derived"] is True)
                    else "background: white !important; color: a9a9a9"
                )
                if first:
                    html.append(
                        f'<tr style="border-top: 1px solid black; {style}"><th rowspan="{len(component.parameters)}" style="vertical-align:top; color:black">{title}</th>'
                    )
                    html.append(
                        f'<td>{p["key"]}</td><td>{p["value"]}</td><td>{p["unit"]}</td>'
                    )
                    first = False
                else:
                    html.append('<tr style="{}">'.format(style))
                    html.append(
                        f'<td>{p["key"]}</td><td>{p["value"]}</td><td>{p["unit"]}</td>'
                    )
                html.append("</tr>")

        append_component(html, self, self.type)
        for component in self.sub_components:
            append_component(html, component, component.type)

        html.append("</table>")
        return "\n".join(html)
