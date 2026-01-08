#!/usr/bin/env python3
"""
LifeCore V3 - Event System
==========================

Système d'événements pour réagir aux situations imprévues.
Permet aux LifeCore de recevoir et émettre des événements asynchrones.

Types d'événements:
- Alert: Situation critique (panne, batterie faible)
- Warning: Attention requise (retard, capacité proche limite)
- Info: Information (livraison terminée, objectif atteint)
- Demand: Nouvelle demande/commande

Usage:
    from lifecore.event import EventBus, Event, Alert, EventType
    
    bus = EventBus()
    bus.subscribe("battery_low", lambda e: handle_battery(e))
    bus.emit(Alert("battery_low", severity=0.9, source="drone_42"))
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import time


class EventType(Enum):
    """Types d'événements."""
    INFO = "info"           # Information
    WARNING = "warning"     # Attention requise
    ALERT = "alert"         # Critique
    DEMAND = "demand"       # Nouvelle demande
    FAILURE = "failure"     # Panne
    RECOVERY = "recovery"   # Retour à la normale
    CHANGE = "change"       # Changement de contexte


class Severity(Enum):
    """Niveaux de sévérité."""
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    CRITICAL = 0.9


@dataclass
class Event:
    """Événement de base."""
    name: str
    event_type: EventType = EventType.INFO
    severity: float = 0.5  # 0-1
    source: str = ""
    target: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    handled: bool = False
    
    def __hash__(self):
        return hash((self.name, self.timestamp, self.source))


@dataclass
class Alert(Event):
    """Alerte critique."""
    event_type: EventType = EventType.ALERT
    severity: float = 0.8
    requires_action: bool = True
    timeout_seconds: float = 60.0


@dataclass
class Demand(Event):
    """Nouvelle demande."""
    event_type: EventType = EventType.DEMAND
    priority: int = 5
    deadline_seconds: float = 3600.0
    quantity: float = 1.0


@dataclass
class Failure(Event):
    """Panne ou dysfonctionnement."""
    event_type: EventType = EventType.FAILURE
    severity: float = 0.9
    affected_capacity: float = 0.0  # % de capacité perdue
    estimated_recovery: float = 0.0  # Temps estimé de récupération


# === EVENT HANDLERS ===

EventHandler = Callable[[Event], None]


class EventBus:
    """Bus d'événements central.
    
    Permet la communication asynchrone entre LifeCores.
    Pattern publish/subscribe.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._type_subscribers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._history: List[Event] = []
        self._pending: List[Event] = []
        self._max_history = 1000
    
    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """S'abonne à un événement par nom."""
        self._subscribers[event_name].append(handler)
    
    def subscribe_type(self, event_type: EventType, handler: EventHandler) -> None:
        """S'abonne à tous les événements d'un type."""
        self._type_subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Se désabonne d'un événement."""
        if handler in self._subscribers[event_name]:
            self._subscribers[event_name].remove(handler)
    
    def emit(self, event: Event) -> int:
        """Émet un événement.
        
        Returns:
            Nombre de handlers qui ont traité l'événement
        """
        handlers_called = 0
        
        # Ajouter à l'historique
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        # Handlers par nom
        for handler in self._subscribers.get(event.name, []):
            handler(event)
            handlers_called += 1
        
        # Handlers par type
        for handler in self._type_subscribers.get(event.event_type, []):
            handler(event)
            handlers_called += 1
        
        if handlers_called > 0:
            event.handled = True
        else:
            self._pending.append(event)
        
        return handlers_called
    
    def get_pending(self, event_type: Optional[EventType] = None) -> List[Event]:
        """Récupère les événements non traités."""
        if event_type:
            return [e for e in self._pending if e.event_type == event_type]
        return list(self._pending)
    
    def get_history(self, 
                    event_type: Optional[EventType] = None,
                    since: Optional[float] = None,
                    limit: int = 100) -> List[Event]:
        """Récupère l'historique des événements."""
        result = self._history
        
        if event_type:
            result = [e for e in result if e.event_type == event_type]
        
        if since:
            result = [e for e in result if e.timestamp >= since]
        
        return result[-limit:]
    
    def clear_pending(self) -> int:
        """Vide les événements pending."""
        count = len(self._pending)
        self._pending.clear()
        return count


# === EVENT-AWARE MIXIN ===

class EventAwareMixin:
    """Mixin pour rendre un LifeCore conscient des événements.
    
    Ajoute:
    - Référence au bus d'événements
    - Méthodes pour émettre/recevoir
    - Réactions automatiques aux alertes
    """
    
    _event_bus: Optional[EventBus] = None
    _event_handlers: Dict[str, List[EventHandler]] = {}
    
    def set_event_bus(self, bus: EventBus) -> None:
        """Connecte ce LifeCore à un bus d'événements."""
        self._event_bus = bus
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Configure les handlers par défaut."""
        if not self._event_bus:
            return
        
        # Handler pour les alertes critiques
        self._event_bus.subscribe_type(
            EventType.ALERT, 
            self._on_alert
        )
        
        # Handler pour les pannes
        self._event_bus.subscribe_type(
            EventType.FAILURE,
            self._on_failure
        )
    
    def emit_event(self, event: Event) -> int:
        """Émet un événement sur le bus."""
        if self._event_bus:
            # Ajouter la source si non spécifiée
            if not event.source and hasattr(self, 'name'):
                event.source = getattr(self, 'name', 'unknown')
            return self._event_bus.emit(event)
        return 0
    
    def emit_alert(self, name: str, severity: float = 0.8, **data) -> int:
        """Raccourci pour émettre une alerte."""
        return self.emit_event(Alert(
            name=name,
            severity=severity,
            source=getattr(self, 'name', 'unknown'),
            data=data
        ))
    
    def emit_demand(self, name: str, priority: int = 5, 
                   deadline: float = 3600, **data) -> int:
        """Raccourci pour émettre une demande."""
        return self.emit_event(Demand(
            name=name,
            priority=priority,
            deadline_seconds=deadline,
            data=data
        ))
    
    def _on_alert(self, event: Alert) -> None:
        """Réaction par défaut aux alertes.
        
        Override pour personnaliser.
        """
        pass
    
    def _on_failure(self, event: Failure) -> None:
        """Réaction par défaut aux pannes.
        
        Override pour personnaliser.
        """
        pass


# === BUILT-IN EVENTS ===

def battery_low_alert(source: str, level: float) -> Alert:
    """Crée une alerte batterie faible."""
    return Alert(
        name="battery_low",
        severity=1.0 - level,  # Plus la batterie est basse, plus c'est critique
        source=source,
        data={"battery_level": level}
    )


def capacity_warning(source: str, current: float, max_capacity: float) -> Event:
    """Crée un warning de capacité proche limite."""
    utilization = current / max_capacity if max_capacity > 0 else 1.0
    return Event(
        name="capacity_warning",
        event_type=EventType.WARNING,
        severity=utilization,
        source=source,
        data={"current": current, "max": max_capacity, "utilization": utilization}
    )


def delivery_complete(source: str, order_id: int, delivery_time: float) -> Event:
    """Crée un événement de livraison terminée."""
    return Event(
        name="delivery_complete",
        event_type=EventType.INFO,
        severity=0.0,
        source=source,
        data={"order_id": order_id, "delivery_time": delivery_time}
    )


def equipment_failure(source: str, component: str, 
                      affected_capacity: float = 1.0,
                      estimated_recovery: float = 0.0) -> Failure:
    """Crée un événement de panne équipement."""
    return Failure(
        name="equipment_failure",
        source=source,
        affected_capacity=affected_capacity,
        estimated_recovery=estimated_recovery,
        data={"component": component}
    )


def demand_spike(source: str, current_demand: float, 
                normal_demand: float) -> Event:
    """Crée un événement de pic de demande."""
    ratio = current_demand / normal_demand if normal_demand > 0 else 1.0
    return Event(
        name="demand_spike",
        event_type=EventType.WARNING if ratio < 2 else EventType.ALERT,
        severity=min(1.0, (ratio - 1) / 3),  # Critical si 4x normal
        source=source,
        data={"current": current_demand, "normal": normal_demand, "ratio": ratio}
    )


# === EVENT AGGREGATOR ===

class EventAggregator:
    """Agrège plusieurs événements similaires.
    
    Évite le flood d'événements identiques.
    """
    
    def __init__(self, time_window: float = 60.0, count_threshold: int = 3):
        self.time_window = time_window
        self.count_threshold = count_threshold
        self._event_counts: Dict[str, List[float]] = defaultdict(list)
    
    def should_emit(self, event_name: str) -> bool:
        """Retourne True si l'événement doit être émis.
        
        Filtre les événements trop fréquents.
        """
        now = time.time()
        
        # Nettoyer les anciens timestamps
        self._event_counts[event_name] = [
            t for t in self._event_counts[event_name]
            if now - t < self.time_window
        ]
        
        # Vérifier le seuil
        if len(self._event_counts[event_name]) >= self.count_threshold:
            return False
        
        # Enregistrer ce timestamp
        self._event_counts[event_name].append(now)
        return True
    
    def get_counts(self) -> Dict[str, int]:
        """Retourne le nombre d'occurrences par événement."""
        now = time.time()
        return {
            name: len([t for t in times if now - t < self.time_window])
            for name, times in self._event_counts.items()
        }
