"""Interface courtier neutre. Aucun ordre n'est envoyé sans connecteur identifié."""
from dataclasses import dataclass
from typing import Protocol
@dataclass
class AccountSnapshot:
 cash: float
 currency: str
 buying_power: float|None=None
class BrokerAdapter(Protocol):
 def get_account(self)->AccountSnapshot: ...
 def get_positions(self)->list[dict]: ...
 def submit_order(self,symbol:str,side:str,quantity:float,order_type:str,limit_price:float|None=None)->dict: ...
class ReadOnlyBrokerAdapter:
 name="Courtier non configuré"
 def get_account(self): raise NotImplementedError("Courtier non configuré")
 def get_positions(self): raise NotImplementedError("Courtier non configuré")
 def submit_order(self,*args,**kwargs): raise PermissionError("Ordres désactivés : courtier non identifié et aucune autorisation explicite.")
